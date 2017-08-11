from enum import Enum
import ctypes
import numpy
import sys

import ufl
import firedrake

from pyop2 import op2
from pyop2 import compilation
from pyop2 import base as pyop2
from pyop2 import sequential as seq
from pyop2.datatypes import IntType, ScalarType, as_cstr, as_ctypes
from pyop2.utils import get_petsc_dir

from .kernels import projection_kernel


__all__ = ("projection_wrapper", "ProjectionMode")


class JITModule(seq.JITModule):

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        # No caching
        return None


class ProjectionMode(Enum):
    to_firedrake = 1
    from_firedrake = 2
    compute_scaling = 3


def projection_wrapper(expr, quadrature_rule, mode):
    assert mode in ProjectionMode
    assert isinstance(expr, firedrake.Function)

    V = expr.function_space()
    domain = expr.ufl_domain()
    if mode == ProjectionMode.to_firedrake:
        kernel = projection_kernel(firedrake.TestFunction(V), quadrature_rule)
        input = op2.Global(V.value_size, dtype=expr.dat.dtype)(op2.READ)
        output = expr.dat(op2.INC, expr.cell_node_map()[op2.i[0]])
    elif mode == ProjectionMode.from_firedrake:
        kernel = projection_kernel(expr, quadrature_rule)
        input = expr.dat(op2.READ, expr.cell_node_map())
        output = op2.Global(V.value_size, dtype=expr.dat.dtype)(op2.INC)
    else:
        kernel = projection_kernel(ufl.classes.QuadratureWeight(domain), quadrature_rule)
        input = None
        output = op2.Global(1, dtype=ScalarType)(op2.INC)

    kernel = op2.Kernel(kernel, kernel.name)
    dim = domain.geometric_dimension()

    xp = op2.Global(dim, dtype=ScalarType)(op2.READ)
    lambda_ = op2.Global(1, dtype=ScalarType)(op2.READ)
    coords = domain.coordinates.dat(op2.READ, domain.coordinates.cell_node_map())

    if input is not None:
        args = [output, coords, input]
    else:
        args = [output, coords]

    args.extend([xp, lambda_])
    for i, arg in enumerate(args):
        arg.position = i

    itspace = pyop2.build_itspace(args, op2.Subset(domain.cell_set, [0]))

    mod = JITModule(kernel, itspace, *args, delay=True)
    return mod.code_to_compile, mod._wrapper_name


template = """
#include <stddef.h>
#include <stdio.h>
#include <inttypes.h>
#include <spatialindex/capi/sidx_api.h>

{wrapper_code}

void mesh_loop(
    {IntType} Nparticle,
    void *sidx,
    const {ScalarType} *Xhat,
    const {ScalarType} *lambda,
    const {ScalarType} *coords,
    const {IntType} *coords_map,
    {read_write_args})
{{
    int64_t *ids = NULL;
    {IntType} *cells = NULL;
    uint64_t ncell = 0;
    {IntType} nwarning = 0;
    RTError err;
    PetscErrorCode ierr;
    {ScalarType} xmin[{dimension}];
    {ScalarType} xmax[{dimension}];
    for ( {IntType} p = 0; p < Nparticle; p++ ) {{
        const {ScalarType} lambdap = lambda[p];
        for ( int k = 0; k < {dimension}; k++ ) {{
            xmin[k] = Xhat[p*{dimension} + k] - lambdap;
            xmax[k] = Xhat[p*{dimension} + k] + lambdap;
        }}
        err = Index_Intersects_id(sidx, xmin, xmax, {dimension}, &ids, &ncell);

        if ( err != RT_None) {{
            nwarning++;
            /* Assume if we didn't locate that we just have zero */
            continue;
        }}

        ierr = PetscMalloc1(ncell, &cells); CHKERRV(ierr);
        for ( size_t i = 0; i < ncell; i++ ) {{
            cells[i] = ({IntType})ids[i];
        }}
        {initialise_uhat}
        {call_wrapper}
        free(ids);
        ierr = PetscFree(cells); CHKERRV(ierr);
    }}
    if (nwarning > 0) {{
        fprintf(stderr, "WARNING: Failed to locate %d particles\\n", nwarning);
    }}
}}
"""

initialise_uhat = """for ( int k = 0; k < {value_size}; k++ ) {{
            uhat[p*{value_size} + k] = 0;
        }}"""


def projection_loop(u, quadrature_rule, mode):
    wrapper_code, wrapper_name = projection_wrapper(u, quadrature_rule, mode)
    config = {"wrapper_code": wrapper_code,
              "wrapper_name": wrapper_name,
              "IntType": as_cstr(IntType),
              "ScalarType": as_cstr(ScalarType),
              "value_size": numpy.prod(u.ufl_shape, dtype=int),
              "dimension": u.ufl_domain().geometric_dimension()}

    if mode == ProjectionMode.to_firedrake:
        read_write_args = """const {ScalarType} *uhat,
        {ScalarType} *u,
        {IntType} *u_map"""
        call_wrapper = """{wrapper_name}(0, ncell, cells,
                           u,
                           u_map,
                           coords,
                           coords_map,
                           &uhat[p*{value_size}],
                           &Xhat[p*{dimension}],
                           &lambda[p]);"""
        config["initialise_uhat"] = ""
    elif mode == ProjectionMode.from_firedrake:
        read_write_args = """{ScalarType} *uhat,
        const {ScalarType} *u,
        const {IntType} *u_map""".format(**config)
        call_wrapper = """{wrapper_name}(0, ncell, cells, &uhat[p],
                           coords,
                           coords_map,
                           u,
                           u_map,
                           &Xhat[p*{dimension}],
                           &lambda[p]);"""
        config["initialise_uhat"] = initialise_uhat.format(**config)
    else:
        read_write_args = """{ScalarType} *uhat"""
        call_wrapper = """{wrapper_name}(0, ncell, cells, &uhat[p],
                       coords,
                       coords_map,
                       &Xhat[p*{dimension}],
                       &lambda[p]);""".format(**config)
        config["initialise_uhat"] = initialise_uhat.format(**config)
    config["call_wrapper"] = call_wrapper.format(**config)
    config["read_write_args"] = read_write_args.format(**config)

    argtypes = [as_ctypes(IntType),  # nparticle
                ctypes.c_voidp,      # spatialindex
                ctypes.c_voidp,      # xhat
                ctypes.c_voidp,      # lambda
                ctypes.c_voidp,      # coords
                ctypes.c_voidp,      # coords-map
                ctypes.c_voidp]      # uhat

    if mode != ProjectionMode.compute_scaling:
        argtypes.extend([ctypes.c_voidp,   # u
                         ctypes.c_voidp])  # u-map

    cppargs = ["-I%s/include" % d for d in get_petsc_dir()] + ["-I%s/include" % sys.prefix]
    ldargs = (["-L%s/lib" % d for d in get_petsc_dir()] +
              ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()] +
              ["-lpetsc", "-lm"] +
              ["-L%s/lib" % sys.prefix, "-lspatialindex_c", "-Wl,-rpath,%s/lib" % sys.prefix])
    return compilation.load(template.format(**config), "c",
                            "mesh_loop",
                            cppargs=cppargs,
                            ldargs=ldargs,
                            restype=None,
                            argtypes=argtypes,
                            comm=u.comm)
