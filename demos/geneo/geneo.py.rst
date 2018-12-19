==============================
 Demo of GenEO preconditioner
==============================

::

  from firedrake import *
  from firedrake import dmplex
  import ufl
  from firedrake.petsc import PETSc
  from slepc4py import SLEPc
  from mpi4py import MPI
  import numpy
  from pyop2.datatypes import IntType
  from collections import defaultdict
  from functools import reduce
  mesh = UnitSquareMesh(100, 100, distribution_parameters={"overlap_type":
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

::

  eps.interpolate(conditional(ufl.And(y > 0.2, y < 0.4), Constant(1e6),
                              conditional(ufl.And(y > 0.6, y < 0.8), Constant(1e5),
                                          Constant(1))))
  # eps.interpolate(conditional(expr, 1000, 1))
  
  
  a = (eps*dot(grad(u), grad(v)))*dx
  
  bcs = DirichletBC(V, 0, 4)
  
  
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
      iset = PETSc.IS().createGeneral(leafdata, comm=COMM_SELF)
      return iset
  
  
  class GeneoPC(PCBase):
  
      def initialize(self, pc):
          A, P = pc.getOperators()
          ctx = P.getPythonContext()
          if V.value_size > 1:
              raise NotImplementedError
  
          multiplicities = build_multiplicities(V)
          intersections = build_intersections(V)
          P = assemble(ctx.a, bcs=ctx.row_bcs, mat_type="is").M.handle
  
          bcs, = ctx.row_bcs
          nodes = bcs.nodes
  
          # Disgusting hack. We put 1 on the diagonal with
          # INSERT_VALUES, but when doing MatConvert, the MatIS has
          # forgotten this, so Dirichlet nodes shared across N processes
          # get "N" on the diagonal, rather than 1. By putting 1/N on
          # the diagonal on each process, the global matrix is "right".
          if len(nodes) > 0:
              P.setValuesLocalRCV(nodes.reshape(-1, 1),
                                  nodes.reshape(-1, 1),
                                  (1/multiplicities.indices[nodes]).reshape(-1, 1),
                                  addv=PETSc.InsertMode.INSERT_VALUES)
          P.assemble()
  
          ipc = PETSc.PC().create(comm=pc.comm)
          ipc.setOptionsPrefix("geneo_")
          ipc.setOperators(P, P)
          ipc.setType("geneo")
  
          dmplex.setupgeneopc(ipc, multiplicities, intersections)
          ipc.setFromOptions()
          ipc.incrementTabLevel(1, parent=pc)
          self.ipc = ipc
  
          # Aneu = P.getISLocalMat()
  
          # Passembled = P.copy()
          # Passembled = Passembled.convert(PETSc.Mat.Type.AIJ)
          # Adir = assemble(ctx.a, bcs=ctx.row_bcs, mat_type="aij").M.handle
  
          # PETSc.Sys.Print("||Passembled - Adir||: %s" % (Passembled - Adir).norm())
          # lgmap = V.dof_dset.lgmap
  
          # ises = (PETSc.IS().createGeneral(lgmap.indices, comm=COMM_SELF), )
          # Adir, = Adir.createSubMatrices(ises, ises)
  
          # Dj = PETSc.Vec().create(comm=COMM_SELF)
          # Dj.setSizes(multiplicities.getLocalSize())
          # Dj.setUp()
          # Dj.array[:] = multiplicities.indices
          # Dj.reciprocal()
  
          # DAdirD = Adir.copy()
          # DAdirD.diagonalScale(Dj, Dj)
  
          # eps = SLEPc.EPS().create(comm=COMM_SELF)
          # eps.setOperators(Aneu, DAdirD)
          # eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
  
          # eps.setType(eps.Type.ARPACK)
          # eps.setWhichEigenpairs(eps.Which.SMALLEST_MAGNITUDE)
          # eps.setDimensions(nev=5)
          # eps.setFromOptions()
  
          # eps.solve()
  
          # nconv = eps.getConverged()
  
          # PETSc.Sys.syncPrint("[%d]: converged %d eigenvalues" % (pc.comm.rank, nconv),
          #                     comm=pc.comm)
          # for i in range(nconv):
          #     PETSc.Sys.syncPrint("[%d]: Eigenvalue %d has value %s" % (pc.comm.rank, i, eps.getEigenvalue(i)),
          #                         comm=pc.comm)
          # PETSc.Sys.syncFlush(comm=pc.comm)
  
          # Adirgeneo = PETSc.Mat().create(comm=COMM_SELF)
          # Aneugeneo = PETSc.Mat().create(comm=COMM_SELF)
          # DAdirDgeneo = PETSc.Mat().create(comm=COMM_SELF)
          # for mat, pattern in zip([Aneugeneo, Adirgeneo, DAdirDgeneo],
          #                         ["debug%d.setup.Aneu.bin" % (pc.comm.rank),
          #                          "debug%d.setup.ADir.bin" % (pc.comm.rank),
          #                          "debug%d.setup.DADirD.bin" % (pc.comm.rank)
          #                         ]):
          #     vwr = PETSc.Viewer().create(comm=COMM_SELF)
          #     vwr.setType(vwr.Type.BINARY)
          #     vwr.setFileMode(vwr.Mode.READ)
          #     vwr.setFileName(pattern)
          #     mat.load(viewer=vwr)
  
          # numpy.set_printoptions(linewidth=255)
          # for name, gmat, mat in zip(["Aneu", "Adir", "DAdirD"],
          #                            [Aneugeneo, Adirgeneo, DAdirDgeneo],
          #                            [Aneu, Adir, DAdirD]):
          #     diff = gmat - mat
          #     PETSc.Sys.syncPrint("[%d]: %s ||A - E|| %s" % (pc.comm.rank, name, diff.norm()),
          #                         comm=pc.comm)
          #     if diff.norm() > 1e-5:
          #         print("geneo mat")
          #         print("=========")
          #         print(gmat[:, :])
          #         print("my    mat")
          #         print("=========")
          #         print(mat[:, :])
          #         print("diff  mat")
          #         print("=========")
          #         print((gmat - mat)[:, :])
          # PETSc.Sys.syncFlush(comm=pc.comm)
  
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
  
  uh.rename("Solution")
  eps.rename("Coefficient")
  
  decomp = Function(eps.function_space())
  decomp.assign(Constant(mesh.comm.rank))
  decomp.rename("Decomposition")
  File("solution.pvd").write(uh, eps, decomp)
