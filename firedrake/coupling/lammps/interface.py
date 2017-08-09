import numpy
import firedrake
from .quadrature import make_quadrature
from .codegen import projection_loop, ProjectionMode

__all__ = ("project", "ParticleProperties", "ParticleData")


class ParticleProperties(object):
    """Wrap up LAMMPS particle properties.

    :arg x: numpy array of positions, with shape (nparticle,
        geometric_dimension)
    :arg lambda_: numpy array of interaction radiuses, with shape
        (nparticle, )
    """
    def __init__(self, x, lambda_):
        self.x = x
        self.lambda_ = lambda_
        size, = lambda_.shape
        xsize, dim = x.shape
        assert size == xsize
        self.size = size
        self.dimension = dim


class ParticleData(object):
    """Wrap up LAMMPS particle data.

    :arg properties: a :class:`ParticleProperties` object.
    :arg value: a numpy array of values, with shape (nparticle,
        ncomponent)
    """
    def __init__(self, value, properties):
        self.properties = properties
        self.value = value
        size, value_size = value.shape
        assert size == self.properties.size
        self.value_size = value_size


def project(donor, target, kernel_density):
    """Project data between Firedrake and LAMMPS.

    :arg donor: donor value.
    :arg target: target value.
    :arg kernel_density: function that provides a symmetric kernel
       density on the reference n-ball.  Given an (N, dim) array of
       quadrature points, it should return appropriate weights.

    There are two projection modes.

    1. If ``donor`` is a Firedrake :class:`~.Function`, then an L^2
       projection into ``target`` (a :class:`ParticleData` object) is
       performed using a trivial mass matrix.  Note, this is *only*
       suitable for transferring intensive quantities (e.g. density)
       from Firedrake to LAMMPS.

    2. If ``donor`` is a :class:`ParticleData`, then an L^2 projection
       into ``target`` is performed by inverting a standard finite
       element mass matrix.  Note this is *only* suitable for
       transferring extensive quantities (e.g. mass) from LAMMPS to
       Firedrake.
    """
    if isinstance(donor, firedrake.Function):
        assert isinstance(target, ParticleData), "Target must be ParticleData"
        # Firedrake->LAMMPS
        domain = donor.ufl_domain()
        f2l = True
        V = donor.function_space()
        assert V.value_size == target.value_size, "Mismatching value sizes"
    else:
        assert isinstance(donor, ParticleData), "Donor must be ParticleData"
        assert isinstance(target, firedrake.Function), "Target must be Function"
        # LAMMPS->FIREDRAKE
        domain = target.ufl_domain()
        f2l = False
        V = target.function_space()
        assert V.value_size == donor.value_size, "Mismatching value sizes"

    if domain.geometric_dimension() != domain.topological_dimension():
        raise NotImplementedError("Not for embedded manifolds")

    quadrature = make_quadrature(kernel_density, domain.geometric_dimension())

    # Denote LAMMPS quantities by _p, Firedrake by _f, the kernel
    # density at x_p, lambda_p by f.
    # To map extensive quantities (e.g. mass) from LAMMPS->Firedrake
    # We do an L^2 projection, testing with v \in V
    # \int u_f v dx = \int u_p f(x_p, lambda_p) v dx
    #
    # So first we assemble the right hand side residual by testing the
    # "smeared" particles with a normal Firedrake test function and
    # then inverting the mass matrix onto it as usual.
    #
    # If one wanted a smoother projection, one could replace this with
    # an H^1 projection, requiring that we can compute grad f.

    # To map intensive quantities (e.g. density) from
    # Firedrake->LAMMPS
    # We do an L^2 projection testing with f.
    # \int u_p f dx = \int u_f f dx
    # Since the LAMMPS basis functions are one-dimensional, the mass
    # inverse is trivial, it is just scaling by \int f dp.
    # Normally, this is one (the kernel density is normalised), but
    # where part of the particle radius falls out of the domain, we
    # must scale by a an appropriate weighting.
    if f2l:
        funptr = projection_loop(donor, quadrature, ProjectionMode.from_firedrake)

        scaling = projection_loop(donor, quadrature, ProjectionMode.compute_scaling)
        nparticle = target.properties.size
        assert target.properties.dimension == domain.geometric_dimension()
        mass = numpy.zeros_like(target.properties.lambda_)

        for d in [domain.coordinates, donor]:
            # We're going to read from these.
            d.dat._force_evaluation(read=True, write=False)

        scaling(nparticle, domain.spatial_index.ctypes,
                target.properties.x.ctypes.data,
                target.properties.lambda_.ctypes.data,
                domain.coordinates.dat._data.ctypes.data,
                domain.coordinates.cell_node_map().values_with_halo.ctypes.data,
                mass.ctypes.data)

        funptr(nparticle, domain.spatial_index.ctypes,
               target.properties.x.ctypes.data,
               target.properties.lambda_.ctypes.data,
               domain.coordinates.dat._data.ctypes.data,
               domain.coordinates.cell_node_map().values_with_halo.ctypes.data,
               target.value.ctypes.data,
               donor.dat._data.ctypes.data,
               donor.cell_node_map().values_with_halo.ctypes.data)
        target.value /= mass.reshape(-1, 1)
        return target
    else:
        funptr = projection_loop(target, quadrature, ProjectionMode.to_firedrake)
        nparticle = donor.properties.size
        assert donor.properties.dimension == domain.geometric_dimension()
        residual = firedrake.Function(V)
        domain.coordinates.dat._force_evaluation(read=True, write=False)
        funptr(nparticle, domain.spatial_index.ctypes,
               donor.properties.x.ctypes.data,
               donor.properties.lambda_.ctypes.data,
               domain.coordinates.dat._data.ctypes.data,
               domain.coordinates.cell_node_map().values_with_halo.ctypes.data,
               donor.value.ctypes.data,
               residual.dat._data.ctypes.data,
               residual.cell_node_map().values_with_halo.ctypes.data)
        A = firedrake.assemble(firedrake.inner(firedrake.TestFunction(V),
                                               firedrake.TrialFunction(V))*firedrake.dx)
        firedrake.solve(A, target, residual)
        return target
