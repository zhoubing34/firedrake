from finat.quadrature import QuadratureRule
from finat.point_set import PointSet
try:
    import quadpy
except ImportError as e:
    raise RuntimeError("Projection between lammps requires the 'quadpy' package") from e


__all__ = ("make_quadrature", )


def make_quadrature(f, dim):
    """Create a quadrature rule of degree 5 for the dim-ball.

    :arg f: kernel density function.
    :arg dim: dimension.
    :returns: :class:`QuadratureRule` which integrates to the volume of the unit dim-ball.
    """
    quad = quadpy.nball.Dobrodeev1978(dim)
    points = quad.points
    weights = quad.weights
    weights = weights * f(points)
    weights = weights / weights.sum()
    return QuadratureRule(PointSet(points), weights)


