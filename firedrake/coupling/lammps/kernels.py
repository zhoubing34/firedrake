from functools import reduce

import numpy

import ufl
from ufl.corealg.map_dag import map_expr_dag

from coffee import base as ast

import gem
import gem.impero_utils as impero_utils

import tsfc
import tsfc.kernel_interface.firedrake as firedrake_interface
import tsfc.ufl_utils as ufl_utils

from tsfc.coffee import SCALAR_TYPE
from tsfc.finatinterface import create_element


__all__ = ("projection_kernel", )


def jacobian_inverse(domain):
    Jinv = ufl.JacobianInverse(domain)

    C = ufl_utils.coordinate_coefficient(domain)

    expr = ufl_utils.preprocess_expression(Jinv)
    expr = ufl_utils.replace_coordinates(expr, C)
    expr = ufl_utils.simplify_abs(expr)

    builder = firedrake_interface.KernelBuilderBase()
    builder._coefficient(C, "coords")

    context = tsfc.fem.PointSetContext(
        interface=builder,
        ufl_cell=domain.ufl_cell(),
        quadrature_degree=0,
    )
    translator = tsfc.fem.Translator(context)
    return map_expr_dag(translator, expr)


# TODO use fiat_cell.contains_point somehow.
def inside(point, cell, dim_index):
    """Return a predicate determining if a point is in a cell.

    :arg point: gem expression for the point.
    :arg cell: UFL reference cell.
    :arg dim_index: gem Index that runs over the geometric dimension
        of the point expression.
    :returns: logical gem expression.
    """
    if not cell.is_simplex():
        raise NotImplementedError("geometric predicate not supported on hypercubes")
    inranges = []
    dim = cell.geometric_dimension()
    for d in range(dim):
        inrange = gem.LogicalAnd(gem.Comparison(">=", gem.Indexed(point, (d, )),
                                                gem.Literal(0.0)),
                                 gem.Comparison("<=", gem.Indexed(point, (d, )),
                                                gem.Literal(1.0)))
        inranges.append(inrange)

    inrange = reduce(gem.LogicalAnd, inranges)

    return gem.LogicalAnd(gem.Comparison("<=", gem.index_sum(gem.Indexed(point, (dim_index, )), (dim_index, )),
                                         gem.Literal(1)),
                          inrange)


def reference_point(point_set, coords, domain):
    """Map a point_set to reference space.

    :arg point_set: a point set of points in the reference :math:`[-1, 1]^d`
       hypercube.
    :arg coords: gem expression for the global coordinates.
    :arg domain: ufl Mesh.
    :returns: pair of dimension free index and transformed coordinates."""
    dim = domain.geometric_dimension()
    i = gem.Index(name="i")
    j = gem.Index(name="j")

    Jinv = gem.Indexed(jacobian_inverse(domain), (i, j))
    xi = gem.Indexed(point_set.expression, (j, ))
    lam = gem.Indexed(gem.Variable("lambda", (1, )), (0, ))
    xp = gem.Indexed(gem.Variable("xp", (dim, )), (j, ))
    cell_origin = gem.Indexed(coords, (0, j))

    # Jinv*(xi*lambda + xp - origin)
    return i, gem.ComponentTensor(
        gem.IndexSum(gem.Product(Jinv,
                                 gem.Sum(gem.Sum(gem.Product(xi, lam), xp),
                                         gem.Product(gem.Literal(-1), cell_origin))),
                     (j, )),
        (i, ))


def kernel_body(expr, domain, quadrature_rule):
    assert isinstance(expr, ufl.classes.Expr)

    dim = domain.ufl_cell().topological_dimension()
    if dim != domain.ufl_cell().geometric_dimension():
        raise NotImplementedError("Not for embedded manifolds")

    point_set = quadrature_rule.point_set

    q, = point_set.indices

    kernel_args = []
    builder = firedrake_interface.KernelBuilderBase()
    C = ufl_utils.coordinate_coefficient(domain)
    kernel_args.append(builder._coefficient(C, "coords"))
    cell = domain.ufl_cell()
    dim_index, point_expr = reference_point(point_set, builder.coefficient_map[C], domain)

    inside_predicate = inside(point_expr, cell, dim_index)

    prefix_ordering = (q, )
    index_names = {q: "ip"}
    if isinstance(expr, ufl.Argument):
        element = create_element(expr.ufl_element())
        argument_multiindex = element.get_indices()
        argument_multiindices = (argument_multiindex, )
        argument = True
        prefix_ordering = prefix_ordering + argument_multiindex
        if expr.ufl_shape:
            tensor_indices = argument_multiindex[-len(expr.ufl_shape):]
            var = ast.Decl(SCALAR_TYPE, ast.Symbol("variable", rank=expr.ufl_shape),
                           qualifiers=["const"])
        else:
            tensor_indices = ()
            var = ast.Decl(SCALAR_TYPE, ast.Symbol("variable", rank=(1, )),
                           qualifiers=["const"])
        kernel_args.append(var)

        for i, name in zip(argument_multiindex, "ijklmnopq"):
            index_names[i] = name
    else:
        argument_multiindices = ()
        argument = False
        tensor_indices = tuple(gem.Index() for _ in expr.ufl_shape)
        if isinstance(expr, ufl.Coefficient):
            kernel_args.append(builder._coefficient(expr, "coefficient"))
        else:
            # Used for computing scaling factor for particles
            # partially out of the domain
            assert expr == ufl.classes.IntValue(1)

    config = dict(interface=builder,
                  ufl_cell=cell,
                  precision=15,
                  point_indices=(q, ),
                  point_expr=point_expr,
                  argument_multiindices=argument_multiindices)

    context = tsfc.fem.GemPointContext(**config)

    translator = tsfc.fem.Translator(context)
    result = map_expr_dag(translator, expr)

    result = gem.Indexed(result, tensor_indices)

    if argument:
        R = gem.Variable("R", (numpy.prod(element.index_shape, dtype=int), ))
        R = gem.reshape(R, element.index_shape)
        return_variable = gem.Indexed(R, argument_multiindex)
        return_variable, = gem.optimise.remove_componenttensors([return_variable])
        if expr.ufl_shape:
            var = gem.Indexed(gem.Variable("variable", expr.ufl_shape), tensor_indices)
        else:
            var = gem.Indexed(gem.Variable("variable", (1, )), (0, ))
        result = gem.Product(var, result)
        result_arg = ast.Decl(SCALAR_TYPE, ast.Symbol("R", rank=element.index_shape))
    else:
        if expr.ufl_shape:
            return_variable = gem.Indexed(gem.Variable("R", expr.ufl_shape), tensor_indices)
            result_arg = ast.Decl(SCALAR_TYPE, ast.Symbol("R", rank=expr.ufl_shape))
        else:
            return_variable = gem.Indexed(gem.Variable("R", (1, )), (0, ))
            result_arg = ast.Decl(SCALAR_TYPE, ast.Symbol("R", rank=(1, )))

    kernel_args = [result_arg] + kernel_args
    kernel_args.append(ast.Decl(SCALAR_TYPE, ast.Symbol("xp", rank=(dim, )),
                                qualifiers=["const"]))
    kernel_args.append(ast.Decl(SCALAR_TYPE, ast.Symbol("lambda", rank=(1, )),
                                qualifiers=["const"]))

    result = gem.Product(result, quadrature_rule.weight_expression)

    result = gem.Conditional(inside_predicate, result, gem.Zero())

    result = gem.index_sum(result, (q, ))
    result, = gem.impero_utils.preprocess_gem([result])
    impero_c = impero_utils.compile_gem([(return_variable, result)], prefix_ordering)
    body = tsfc.coffee.generate(impero_c, index_names, 15)

    return builder, kernel_args, body


def projection_kernel(expr, domain, quadrature_rule):
    """Produce a kernel that interpolates V at points using a provided quadrature rule.

    :arg expr: The UFL expression.
    :arg domain: The mesh.
    :arg quadrature_rule: A quadrature-rule for the unit
        :math:`d-\text{ball}` that integrates to one.
    :arg transpose: If True, interpolate from points to basis
        functions.  If False, from basis functions to points.
    """
    builder, kernel_args, body = kernel_body(expr, domain, quadrature_rule)

    return builder.construct_kernel("interpolate_kernel", kernel_args, body)
