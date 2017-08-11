.. default-role:: math

Projecting between LAMMPS and Firedrake
=======================================

LAMMPS has ensembles of particles, or atoms, which we will treat as
being each characterised by a position `x_p`, a radius `\lambda_p`,
and a distribution function `f_p(x)` given by:

.. math::

  f_p(x) = f(x_p, \lambda_p, x).

That is to say, the distribution functions have a common form,
parametrised by the location and radius of the atom. The distribution
functions of different particles are treated as non-interacting. That
is to say, it is assumed that:

.. math::

  \int_\Omega f_{p}(x)\, f_{q}(x)\,\mathrm{d}x = 0 \qquad \forall p\neq q. 

Physical quantities are represented using the distribution functions
as a basis:

.. math::

  u^L = u^L_p\, f_p(x)

In this document we will represent quantities over LAMMPS ensembles
with superscript `L` and Firedrake
:class:`~firedrake.function.Function` s with the superscript `F`.

Projecting a LAMMPS quantity into Firedrake
-------------------------------------------

Let `V` be a Firedrake
:class:`~firedrake.functionspace.FunctionSpace`. Then the Galerkin
projection of `u^L` into `u^F\in V` is given by the solution to:

.. math::

  \int_\Omega u^F v^F \,\mathrm{d}x =  \int_\Omega u^L_p\,f_p v^F \,\mathrm{d}x\qquad \forall v^F\in V.

This has the usual Galerkin optimality property, and by substituting
`v^F=1`, it is obvious that the integral of `u` over the domain is
conserved by this operation.

This operation can be implemented by expanding in terms of the finite element basis:

.. math::

  \int_\Omega \phi_i\, \phi_j \,\mathrm{d}x\ u^F_j=  \int_\Omega u^L_p\,f_p\, \phi_i \,\mathrm{d}x.
  
The left hand side of this is simply the assembly of a mass matrix,
while the right hand side is assembled by a bespoke Firedrake
extension, which returns a CoFunction. The resulting system is then solved.

Projecting a Firedrake field onto a LAMMPS ensemble
---------------------------------------------------

Conversely, suppose we have `u^F\in V` and we wish the corresponding
`u^L` to be as close as possible, in a certain sense, to `u^F`. We can
define a function space over a particle ensemble by:

.. math::

  P = \{u^L\, |\, u^L= u^L_p\,f_p(x) \quad \forall u^L_p\in \mathbb{R}\}.

Then the Galerkin projection of `u^F` into `P` is given by, find
`u^P\in P` such that:

.. math::

  \int_\Omega u^Lv^L \,\mathrm{d}x = \int_\Omega u^Fv^L\,\mathrm{d}x \qquad \forall v^L\in P.

Now, expanding in bases produces:

.. math::

  u^L_p\int_\Omega f_p\,f_q \,\mathrm{d}x = u^F_i\int_\Omega \phi_i\, f_q\,\mathrm{d}x

The right hand side is simply the adjoint of the right hand side in
the projection from LAMMPS into Firedrake, while the left hand side is
a new LAMMPS mass matrix. Note that the presumed orthogonality between
distribution functions makes the mass matrix diagonal, hence the
calculation reduces to:

.. math::

  u^L_p = \frac{u^F_i\displaystyle\int_\Omega \phi_i\, f_p\,\mathrm{d}x}{\displaystyle\int_\Omega f_p\,f_p \,\mathrm{d}x}.

Up to the orthogonality assumption, this is an optimal
projection. However note that `1\notin P`, so there is no expectation
that integrals are preserved.
