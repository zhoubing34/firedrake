==============================
 Demo of GenEO preconditioner
==============================

::

  from firedrake import *
  from firedrake import dmplex
  import ufl
  from firedrake.petsc import PETSc
  from mpi4py import MPI
  import numpy
  from pyop2.datatypes import IntType
  from collections import defaultdict
  from functools import reduce
  mesh = UnitSquareMesh(50, 50, distribution_parameters={"overlap_type":
                                                       (DistributedMeshOverlapType.NONE, 0)})

  V = FunctionSpace(mesh, "P", 1)

  u = TrialFunction(V)

  v = TestFunction(V)

  eps = Function(FunctionSpace(mesh, "DG", 0))

  x, y = SpatialCoordinate(mesh)

  exprs = []
  for i in range(10):
      for j in range(10):
          if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
              exprs.append(reduce(ufl.And, [(x > i / 10), (x < (i+1) / 10), (y > j / 10), (y < (j + 1) / 10)]))

  expr = reduce(ufl.Or, exprs)

  eps.interpolate(conditional(expr, 1000, 1))

  a = (eps*dot(grad(u), grad(v)))*dx

  bcs = DirichletBC(V, 0, "on_boundary")


  def mpi_type(dtype, n):
      try:
          tdict = MPI.__TypeDict__
      except AttributeError:
          tdict = MPI._typedict
      base = tdict[dtype.char]
      if n == 1:
          mtype = base.Dup()
      else:
          mtype = base.Create_contiguous(n)
          mtype.Commit()
      return mtype


  def build_intersections(V):
      sf = V.dm.getDefaultSF()
      selected = numpy.unique(V.cell_node_map().values)
      _, leaves, _ = sf.getGraph()
      nleaves, = leaves.shape
      keep = numpy.arange(nleaves, dtype=IntType)[numpy.in1d(leaves, selected)]
      embed_sf = sf.createEmbeddedLeafSF(keep)
      degrees = embed_sf.computeDegree()

      comm = V.comm

      maxdegree = numpy.asarray([degrees.max()])
      comm.Allreduce(MPI.IN_PLACE, maxdegree, op=MPI.MAX)
      maxdegree, = maxdegree
      leafdata = numpy.full(V.node_set.total_size, comm.rank, dtype=IntType)
      rootdata = numpy.empty(sum(degrees), dtype=IntType)
      datatype = mpi_type(leafdata.dtype, 1)
      embed_sf.gatherBegin(datatype, leafdata, rootdata)
      embed_sf.gatherEnd(datatype, leafdata, rootdata)
      datatype.Free()
      # Now rootdata contains the multiplicity of each owned node in the function space.
      # Now we need to get the ghosted representation.  To do this,
      # since the roots are ragged, we make a square array (maxdegree is
      # bounded, so this is OK).
      ghosted = numpy.full((nleaves, maxdegree), -2, dtype=IntType)
      unrolled_roots = numpy.full((len(degrees), maxdegree), -1, dtype=IntType)
      offset = 0
      for i, degree in enumerate(degrees):
          unrolled_roots[i, :degree] = rootdata[offset:offset+degree]
          ghosted[i, :degree] = rootdata[offset:offset+degree]
          offset += degree

      datatype = mpi_type(unrolled_roots.dtype, maxdegree)
      sf.bcastBegin(datatype, unrolled_roots, ghosted)
      sf.bcastEnd(datatype, unrolled_roots, ghosted)
      datatype.Free()
      # OK, now we know locally for every point we can see (keep) which
      # ranks can also see it.
      rank_nodes = defaultdict(list)
      for node in keep:
          ranks = ghosted[node, :]
          for rank in ranks:
              if rank < 0 or rank == comm.rank:
                  continue
              rank_nodes[rank].append(node)
      intersections = [None]*comm.size

      lgmap = V.dof_dset.lgmap
      if lgmap.block_size > 1:
          raise NotImplementedError()
      for rank, nodes in rank_nodes.items():
          intersections[rank] = PETSc.IS().createGeneral(sorted(lgmap.apply(node)), comm=COMM_SELF)
      for i in range(len(intersections)):
          if intersections[i] is None:
              intersections[i] = PETSc.IS().createGeneral([], comm=COMM_SELF)
      return intersections


  def build_multiplicities(V):
      selected = numpy.unique(V.cell_node_map().values)

      sf = V.dm.getDefaultSF()

      _, leaves, _ = sf.getGraph()
      nleaves, = leaves.shape
      keep = numpy.arange(nleaves, dtype=IntType)[numpy.in1d(leaves, selected)]
      embed_sf = sf.createEmbeddedLeafSF(keep)
      degrees = embed_sf.computeDegree()

      leafdata = numpy.full(nleaves, -1, dtype=degrees.dtype)
      datatype = mpi_type(degrees.dtype, 1)
      embed_sf.bcastBegin(datatype, degrees, leafdata)
      embed_sf.bcastEnd(datatype, degrees, leafdata)
      datatype.Free()
      leafdata = leafdata[selected]
      assert all(leafdata >= 0)
      return PETSc.IS().createGeneral(leafdata, comm=COMM_SELF)


  class GeneoPC(PCBase):

      def initialize(self, pc):
          A, P = pc.getOperators()
          ctx = P.getPythonContext()
          if V.value_size > 1:
              raise NotImplementedError

          P = assemble(ctx.a, bcs=ctx.row_bcs, mat_type="is").M.handle
          ipc = PETSc.PC().create(comm=pc.comm)
          ipc.setOptionsPrefix("geneo_")
          ipc.setOperators(P, P)
          ipc.setType("geneo")

          multiplicities = build_multiplicities(V)
          intersections = build_intersections(V)
          dmplex.setupgeneopc(ipc, multiplicities, intersections)
          ipc.setFromOptions()
          ipc.incrementTabLevel(1, parent=pc)
          self.ipc = ipc

      def update(self, pc):
          pass

      def apply(self, pc, x, y):
          self.ipc.apply(x, y)

      def applyTranspose(self, pc, x, y):
          self.ipc.applyTranspose(x, y)

      def view(self, pc, viewer=None):
          super().view(viewer)
          viewer.printfASCII("GENEO preconditioner:\n")
          self.ipc.view(viewer)


  uh = Function(V)
  solve(a == v*dx, uh, bcs=bcs, options_prefix="",
        solver_parameters={"mat_type": "matfree",
                           "pc_type": "python",
                           "pc_python_type": "__main__.GeneoPC",
                           "ksp_initial_guess_nonzero": True})
