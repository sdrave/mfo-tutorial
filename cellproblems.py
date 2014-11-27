from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import coo_matrix

from pymor.core.interfaces import inject_sid
from pymor.discretizations import StationaryDiscretization
from pymor.domaindescriptions import TorusDomain, CircleDomain
from pymor.domaindiscretizers import discretize_domain_default
from pymor.functions import ConstantFunction
from pymor.grids import TriaGrid
from pymor.la import NumpyVectorSpace, NumpyVectorArray
from pymor.operators.basic import NumpyMatrixBasedOperator
from pymor.operators.constructions import (LincombOperator, IdentityOperator, Concatenation,
                                           VectorOperator, ConstantOperator)
from pymor.operators.cg import DiffusionOperatorP1, L2ProductFunctionalP1
from pymor.reductors.stationary import reduce_stationary_coercive
from pymor.gui.qt import PatchVisualizer, Matplotlib1DVisualizer


class ZeroMeanStationaryDiscretization(StationaryDiscretization):

    def __init__(self, operator, rhs, dirichlet_operator, dirichlet_rhs, mean_value_corrector,
                 products=None, functionals=None, parameter_space=None, estimator=None, visualizer=None,
                 cache_region='disk', name=None):

        super(ZeroMeanStationaryDiscretization, self).__init__(operator, rhs, products=products,
                                                               functionals=functionals, parameter_space=parameter_space,
                                                               estimator=estimator, visualizer=visualizer,
                                                               cache_region=cache_region, name=name)
        self.dirichlet_operator = dirichlet_operator
        self.dirichlet_rhs = dirichlet_rhs
        self.mean_value_corrector = mean_value_corrector

    def _solve(self, mu=None):
        mu = self.parse_parameter(mu)
        self.logger.info('Solving {} for {} ...'.format(self.name, mu))
        U = self.dirichlet_operator.apply_inverse(self.rhs.as_vector(mu=mu), mu=mu)
        return self.mean_value_corrector.apply(U)


def reduce_zero_mean_value_stationary(discretization, RB, coercivity_estimator=None, extends=None):
    functionals = dict(discretization.functionals)
    d = StationaryDiscretization(discretization.operator, discretization.rhs,
                                 products=discretization.products, functionals=functionals,
                                 parameter_space=discretization.parameter_space, cache_region=None,
                                 name=discretization.name)
    return reduce_stationary_coercive(d, RB, coercivity_estimator=coercivity_estimator, extends=extends)


class CellProblemRHSOperator(NumpyMatrixBasedOperator):

    sparse = False
    range = NumpyVectorSpace(1)

    def __init__(self, grid, diffusion_function, dim_ind, name=None):
        self.source = NumpyVectorSpace(grid.size(grid.dim))
        self.grid = grid
        self.diffusion_function = diffusion_function
        self.dim_ind = dim_ind
        self.name = name
        self.build_parameter_type(inherits=(diffusion_function,))

    def _assemble(self, mu=None):
        g = self.grid

        F = - self.diffusion_function(g.centers(0), mu=mu)

        EI = np.zeros(g.dim)
        EI[self.dim_ind] = 1.
        if g.dim == 2:
            SF_GRAD = np.array(([-1., -1.],
                                [1., 0.],
                                [0., 1.]))
        elif g.dim == 1:
            SF_GRAD = np.array(([-1.],
                                [1., ]))
        else:
            raise NotImplementedError

        SF_GRADS = np.einsum('eij,pj->epi', g.jacobian_inverse_transposed(0), SF_GRAD)

        # integrate the products of the function with the shape functions on each element
        # -> shape = (g.size(0), number of shape functions)
        SF_INTS = np.einsum('e,epi,i,e->ep', F, SF_GRADS, EI, g.volumes(0)).ravel()

        # map local DOFs to global DOFs
        SF_I = g.subentities(0, g.dim).ravel()
        I = np.array(coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim))).todense()).ravel()

        return I.reshape((1, -1))


class DirichletOperator(NumpyMatrixBasedOperator):

    def __init__(self, operator):
        self.operator = operator
        self.source = operator.source
        self.range = operator.range
        self.build_parameter_type(inherits=(operator,))

    def _assemble(self, mu=None):
        matrix = self.operator.assemble(mu)._matrix.tolil()
        matrix[0] = 0.
        matrix[0, 0] = 1.
        return matrix.tocsc()


class DirichletFunctional(NumpyMatrixBasedOperator):

    def __init__(self, operator):
        self.operator = operator
        self.source = operator.source
        self.range = operator.range
        self.build_parameter_type(inherits=(operator,))

    def _assemble(self, mu=None):
        matrix = self.operator.assemble(mu)._matrix.copy()
        matrix[0, 0] = 0.
        return matrix


def discretize_cell_problems(diffusion_functions, diffusion_functionals, diameter=1. / 100.):

    dim = diffusion_functions[0].dim_domain
    assert dim in (1, 2)
    assert all(f.dim_domain == dim and f.shape_range == tuple() for f in diffusion_functions)

    if dim == 1:
        domain = CircleDomain([0., 1.])
        grid, boundary_info = discretize_domain_default(domain, diameter=diameter)
        visualizer = Matplotlib1DVisualizer(grid=grid, codim=1)
    else:
        domain = TorusDomain(([0., 0.], [1., 1.]))
        grid, boundary_info = discretize_domain_default(domain, diameter=diameter, grid_type=TriaGrid)
        visualizer = PatchVisualizer(grid=grid, bounding_box=grid.domain, codim=2)
    operators = [DiffusionOperatorP1(grid, boundary_info, diffusion_function=f, name='diffusion_{}'.format(i))
                 for i, f in enumerate(diffusion_functions)]
    operator = LincombOperator(operators, diffusion_functionals)
    dirichlet_operator = DirichletOperator(operator)
    mean_value_functional = L2ProductFunctionalP1(grid, ConstantFunction(1., dim_domain=dim), order=1,
                                                  name='mean_value_functional')
    constant_projection = Concatenation(VectorOperator(NumpyVectorArray(np.ones(grid.size(dim))), copy=False),
                                        mean_value_functional)
    mean_value_corrector = IdentityOperator(constant_projection.source) - constant_projection
    mean_value_corrector.unlock()
    inject_sid(mean_value_corrector, 'cell_problem_mean_value_corrector', grid)
    ones = NumpyVectorArray(np.ones(grid.size(dim)))

    def make_diffusion_integral(f):
        op = ConstantOperator(L2ProductFunctionalP1(grid, f, order=1).apply(ones), source=operator.source)
        op.unlock()
        inject_sid(op, 'cell_problem_diffusion_integral', f, grid)
        return op
    diffusion_integrals = [make_diffusion_integral(f) for f in diffusion_functions]
    diffusion_integral = LincombOperator(diffusion_integrals, diffusion_functionals)

    rhss = []
    for dim_ind in range(dim):
        components = [CellProblemRHSOperator(grid, diffusion_function=f, dim_ind=dim_ind,
                                             name='RHS_Functional_{}_{}'.format(dim_ind, i))
                      for i, f in enumerate(diffusion_functions)]
        rhss.append(LincombOperator(components, diffusion_functionals))

    discretizations = []
    for dim_ind in range(dim):
        rhs = rhss[dim_ind]
        dirichlet_rhs = DirichletFunctional(rhs)
        homogenized_diffusions = [(diffusion_integral - rhss[i] if i == dim_ind else rhss[i] * (-1.))
                                  for i in range(dim)]
        d = ZeroMeanStationaryDiscretization(operator, rhs, dirichlet_operator, dirichlet_rhs,
                                             mean_value_corrector,
                                             functionals={('diffusion', i): f
                                                          for i, f in enumerate(homogenized_diffusions)},
                                             visualizer=visualizer, name='CellProblem_{}'.format(dim_ind))
        discretizations.append(d)

    return discretizations, {'grid': grid, 'boundary_info': boundary_info}
