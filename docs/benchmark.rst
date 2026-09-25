Benchmark
=========

This page benchmarks `rotorch` against `Clifford Group Equivariant Neural Networks
<https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks>`_ (cgenn). Both
implementations are driven by the same example script, so they see the same data, the same
schedule and the same optimizer, and only the layers underneath differ::

    python examples/hulls.py                       # rotorch
    python examples/hulls.py --compile operators   # torch.compile as kingdon's operator wrapper
    python examples/hulls.py --compile model       # torch.compile over the whole model
    python examples/hulls.py --backend triton      # one triton kernel per operator, cuda only
    python examples/hulls.py --impl cgenn          # the original implementation
    python examples/nbody.py --impl fk             # n-body with flash-clifford's layers

The runs below are:

:cgenn: the cgenn implementation, unoptimized.
:cgenn-compiled: the entire cgenn model through :func:`torch.compile`. It has graph breaks,
   since it indexes its weights with a boolean mask whose result has a data dependent shape.
:rotorch: the rotorch example, unoptimized.
:rotorch-operators: hands :func:`torch.compile` to kingdon as its wrapper, so every operator
   kingdon generates for the algebra is compiled on its own.
:rotorch-model: compiles the training step as a whole, fusing across those operators rather
   than stopping at each one. **One graph, no breaks**, and :code:`fullgraph=True` succeeds.
:rotorch-triton: hands kingdon a code printer that emits a triton kernel per operator directly,
   instead of applying :func:`torch.compile` to its torch calls. cuda only.
:rotorch-triton-model: those triton kernels, with the model compiled around them. Stacking the
   two does not help: :code:`--backend triton` and :code:`--compile model` already solve the same
   problem, fusing across operators rather than launching each one, so there is no further
   speed-up to have and this configuration is left out of the tables and figures below.
:fk-eager: n-body only: cgenn's network with the layer of `Flash Clifford
   <https://github.com/maxxxzdn/flash-clifford>`_ (Zhdanov, 2025) in its MLPs, which flash-clifford
   offers as their faster replacement, built by rotorch out of kingdon operators (:code:`--impl fk`). The layer's
   GELU gates and weighted geometric product are one kingdon operator, as they are one kernel in
   flash-clifford. It is a different architecture from cgenn's, so these columns are told apart
   from rotorch's own, though they are measured against cgenn like everything else here.
:fk-triton, fk-model, fk-triton-model: the same, run the way the
   rotorch columns of those names are. Under :code:`--backend triton` that operator is one kernel
   forward and one backward, printed by kingdon rather than written by hand.

Everything here was measured by :code:`examples/sweep.py` on one machine: an **NVIDIA RTX
A4000** (16 GB) beside an **AMD Ryzen Threadripper PRO 7955WX**, Windows 11, torch 2.14 with
cuda 13.0 and python 3.14. Each run is a separate process; each number is the median of 64
steps after eight warm-up ones, and the faster of two repetitions. The batch size goes to 16384
on the card, where 2048 is as far as the cpu is worth taking.

.. tab-set::

   .. tab-item:: Convex hull

      Predict the volume of the convex hull of 16 points in :math:`\mathbb{R}^5`. The algebra is
      :math:`Cl(5)` and the model is a linear embedding followed by four geometric product
      layers. Only 1024 of the 32768 entries of its Cayley tensor are nonzero, so this is the
      example with the most sparsity to spend.

      .. tab-set::

         .. tab-item:: GPU
            :sync: gpu

            .. tab-set::

               .. tab-item:: Time
                  :sync: time

                  .. raw:: html
                     :file: _static/hulls-cuda-time.svg

                  .. raw:: html
                     :file: _static/hulls-cuda-time-table.html

                  Milliseconds per step, and in brackets the speed-up against cgenn at that same
                  batch size. A straight line fitted through all seven points has
                  :math:`r^2 \geq 0.967` for every column.

                  Below batch 512 every column but one sits on a floor between 11 and 29 ms,
                  because a step that small is fixed cost whoever runs it. Eager rotorch's floor
                  is 233 ms, and it holds close to that through batch 2048 (235 ms): a step that
                  barely notices 64 times the data is not doing arithmetic, it is waiting for
                  python to launch its next kernel.

                  Past 2048 the marginal cost sets the order, and :code:`--backend triton` is
                  the fastest column on the card: 2.9 µs per sample against cgenn's 28.7, a
                  tenfold difference in slope, which is the ratio its speedup climbs towards as
                  the fixed cost stops mattering. It is 6.6× at 16384 and has not levelled off.
                  :code:`--compile model` has the lowest floor of all, 10.7 ms, but a slope five
                  times triton's, so it wins below a batch of about 2000 and loses above it: the
                  two are within rounding of each other at exactly 2048 (31.6 ms apiece).

               .. tab-item:: Memory
                  :sync: memory

                  .. raw:: html
                     :file: _static/hulls-cuda-memory.svg

                  .. raw:: html
                     :file: _static/hulls-cuda-memory-table.html

                  Peak MiB of a forward and backward together, and in brackets the fraction of
                  cgenn's peak at that same batch size.

                  This is the cleaner result of the two, because none of it is launch overhead:
                  it is the same 32 blades of every intermediate that the dense einsum writes
                  down and the sparse product never forms. Every ratio drifts down gradually from
                  batch 32 and is close to flat by batch 2048 (0.211× to 0.219× for rotorch across
                  the remaining sizes). At 16384 that is 1.2 to 2.5 GiB against 11.6: cgenn is
                  within 4.4 gigabytes of filling the card and cannot take the next doubling at
                  all, while rotorch is using between a tenth and a fifth of it.

                  Compiling barely moves cgenn's memory, 0.98× of eager. It moves rotorch's more:
                  :code:`--compile operators` reaches 0.13× and :code:`--compile model` 0.10×
                  against eager rotorch's own 0.21×, because inductor fuses away an intermediate
                  the backward turns out not to need.

               .. tab-item:: Parameters
                  :sync: parameters

                  ==============  ==========  ========
                  implementation  parameters  val loss
                  ==============  ==========  ========
                  cgenn           58,849      23.8136
                  rotorch         38,881      23.5201
                  ==============  ==========  ========

                  At batch 16384. All four rotorch columns reach that loss to four decimals and
                  both cgenn columns reach theirs, so neither compiling nor the triton codegen
                  changes what the model computes.

                  Two thirds of the parameters, because the input is a pure vector and the
                  grades only fill in as the products generate them: the first product layer
                  sees grades (0, 1) and needs 5 path weights, the second (0, 1, 2) and 14, the
                  third 45, the fourth 56. cgenn allocates all 56 paths and all six subspace
                  matrices in every layer; the unreachable ones multiply zero coefficients,
                  contributing nothing and receiving no gradient.

               .. tab-item:: Start-up
                  :sync: startup

                  ====================  =======  =======
                  run                   fastest  slowest
                  ====================  =======  =======
                  cgenn                 0.3      1.1
                  cgenn-compiled        2.1      15.0
                  rotorch               0.8      1.0
                  rotorch-operators     26.0     44.2
                  rotorch-model         27.1     43.4
                  rotorch-triton        36.0     185.3
                  ====================  =======  =======

                  Seconds to the first step, over every batch size and repetition.

                  Compilation is paid on that first step and specialized per shape, so each
                  batch size compiles anew, as does the ragged final batch of a validation
                  loader. The spread in the compiled rows is the inductor cache under
                  :code:`$TMPDIR/torchinductor_$USER`, some 60 MB for this model: cold it is
                  tens of seconds, warm it is a few.

                  Triton is the slowest and widest-spread column here, because the tile search
                  compiles a candidate per operator per shape and the autotuner then times the
                  survivors, once per process per batch size: nothing is reused across shapes the
                  way the inductor cache reuses across repetitions. It is still cheap at scale:
                  triton saves 261 ms a step against eager rotorch at batch 16384, so even its
                  slowest cold start of 185 s has paid for itself after about 710 steps.

         .. tab-item:: CPU
            :sync: cpu

            .. tab-set::

               .. tab-item:: Time
                  :sync: time

                  .. raw:: html
                     :file: _static/hulls-cpu-time.svg

                  .. raw:: html
                     :file: _static/hulls-cpu-time-table.html

                  There is no launch overhead on a cpu to hide the sparsity, so eager rotorch
                  shows it without any compiling at all: already a little ahead at batch 32,
                  where a step is mostly fixed cost for both, and 3.0× ahead at 2048. Compiling
                  is worth far more here than on the card, 14.4× at 2048 and 32.8 µs per sample
                  against 517.7, because there is no kernel launch to put a floor under the win.

                  :code:`--backend triton` has no column: there is no triton for the cpu, and
                  kingdon falls back to the torch backend as soon as it sees that no argument is
                  a cuda tensor, so it would be the rotorch column twice.

               .. tab-item:: Memory
                  :sync: memory

                  The sweep weighs a run with :func:`torch.cuda.max_memory_allocated`, which has
                  no cpu counterpart, so there is nothing to report here. The ratios on the card
                  are a property of the algebra rather than the device, and carry over.

               .. tab-item:: Parameters
                  :sync: parameters

                  ==============  ==========  ========
                  implementation  parameters  val loss
                  ==============  ==========  ========
                  cgenn           58,849      23.7728
                  rotorch         38,881      23.4418
                  ==============  ==========  ========

                  At batch 2048, the largest the cpu was taken to.

               .. tab-item:: Start-up
                  :sync: startup

                  =================  =======  =======
                  run                fastest  slowest
                  =================  =======  =======
                  cgenn              0.1      1.2
                  cgenn-compiled     3.6      17.1
                  rotorch            0.6      0.8
                  rotorch-operators  26.4     30.8
                  rotorch-model      27.8     40.8
                  =================  =======  =======

                  Inductor writes C++ for the cpu and needs a compiler for it, so the three
                  compiled rows exist only when one is on the path of the shell that runs the
                  sweep. Earlier attempts recorded in the same csv failed at the first batch
                  size with :code:`no-cpp-compiler` and :code:`no-openmp-headers`, and the sweep
                  skipped the larger ones rather than run the wrong thing.

   .. tab-item:: O(3)

      Regress the determinant of three vectors in 3D, which is a pseudoscalar, so the model has
      to be equivariant under rotations and anti-equivariant under reflections. The algebra is
      :math:`Cl(3)`, eight blades against the convex hull's 32, so there is far less sparsity
      here and the model is small enough that a step is mostly fixed cost either way.

      .. tab-set::

         .. tab-item:: GPU
            :sync: gpu

            .. tab-set::

               .. tab-item:: Time
                  :sync: time

                  .. raw:: html
                     :file: _static/o3-cuda-time.svg

                  .. raw:: html
                     :file: _static/o3-cuda-time-table.html

                  cgenn only reaches 110 ms at batch 16384, so the whole matrix is flatter than
                  the convex hull's and the columns end up within a few milliseconds of each
                  other. What separates them is the floor: :code:`--compile model` steps in 4.4
                  ms where cgenn needs 17.8, and that 4× at small batches is most of the win,
                  since by 16384 every rotorch column has converged on 28 to 32 ms.

                  :code:`Cl(3)` has eight blades against the convex hull's 32, so there is much
                  less sparsity for :code:`--backend triton` to turn into a custom kernel here,
                  and it lands close to eager and compiled-operators rotorch rather than pulling
                  ahead the way it does on the convex hull.

               .. tab-item:: Memory
                  :sync: memory

                  .. raw:: html
                     :file: _static/o3-cuda-memory.svg

                  .. raw:: html
                     :file: _static/o3-cuda-memory-table.html

                  Peak MiB of a forward and backward together. The saving is real but smaller
                  than the convex hull's 0.10× to 0.21×, and for the same reason the timings are
                  closer: eight blades leave less to skip than 32 do. Nothing here comes near
                  filling the card.

               .. tab-item:: Parameters
                  :sync: parameters

                  ==============  ==========  ========
                  implementation  parameters  val loss
                  ==============  ==========  ========
                  cgenn           8,657       0.0124
                  rotorch         4,973       0.0133
                  ==============  ==========  ========

                  At batch 16384, with every column of an implementation on the same loss.

               .. tab-item:: Start-up
                  :sync: startup

                  ====================  =======  =======
                  run                   fastest  slowest
                  ====================  =======  =======
                  cgenn                 0.3      0.4
                  cgenn-compiled        3.7      15.5
                  rotorch               0.3      0.4
                  rotorch-operators     2.9      8.1
                  rotorch-model         3.3      4.1
                  rotorch-triton        13.6     37.9
                  ====================  =======  =======

                  Seconds to the first step. A small model compiles quickly: seconds warm
                  against the minutes the convex hull wants, which is why :code:`--compile
                  model` is worth reaching for here even though the steady-state win is modest.

         .. tab-item:: CPU
            :sync: cpu

            .. tab-set::

               .. tab-item:: Time
                  :sync: time

                  .. raw:: html
                     :file: _static/o3-cpu-time.svg

                  .. raw:: html
                     :file: _static/o3-cpu-time-table.html

                  Eager rotorch is ahead of cgenn at every batch size here, unlike on the card,
                  and :code:`--compile model` steps in 1.9 ms against cgenn's 11.4. Compiling
                  the operators one at a time is slower than not compiling at all: at this size
                  the per-operator call overhead it adds outweighs what each fused kernel saves,
                  which is exactly the case :code:`--compile model` exists to fix.

               .. tab-item:: Memory
                  :sync: memory

                  Not measured on the cpu; see the GPU tab for the ratios.

               .. tab-item:: Parameters
                  :sync: parameters

                  ==============  ==========  ========
                  implementation  parameters  val loss
                  ==============  ==========  ========
                  cgenn           8,657       0.0125
                  rotorch         4,973       0.0081
                  ==============  ==========  ========

                  At batch 2048.

               .. tab-item:: Start-up
                  :sync: startup

                  =================  =======  =======
                  run                fastest  slowest
                  =================  =======  =======
                  cgenn              0.1      0.2
                  cgenn-compiled     5.6      20.1
                  rotorch            0.1      0.2
                  rotorch-operators  4.1      5.7
                  rotorch-model      4.4      4.6
                  =================  =======  =======

   .. tab-item:: N-body

      Predict where five charged particles end up. The algebra is :math:`Cl(3)` again, but the
      model is a graph network over the 20 edges of each five-body system rather than an MLP, so
      there is twenty times more of everything per sample and this is expected to be the one
      example that fills the card. The dataset is simulated by the example rather than read from
      the files the EGNN repository ships, so the physics is cgenn's and the trajectories are
      not. This example also runs the fk columns: flash-clifford says its layer is
      the layer of this network's MLPs made fast, but ships no network to put it in.

      Not swept yet.

   .. tab-item:: Lorentz

      Tell jets from decaying top quarks apart from ordinary ones, in :math:`Cl(1,3)`.

      Not swept yet.

   .. tab-item:: O(5)

      Regress an O(5) invariant of two vectors in :math:`\mathbb{R}^5`. Nearly all of this
      model's 343,125 parameters sit in a plain MLP head that both implementations share, against
      cgenn's 344,077, so it is the example where the two are closest in size and the sparsity
      of the algebra has the least room to matter.

      .. tab-set::

         .. tab-item:: GPU
            :sync: gpu

            .. tab-set::

               .. tab-item:: Time
                  :sync: time

                  .. raw:: html
                     :file: _static/o5-cuda-time.svg

                  .. raw:: html
                     :file: _static/o5-cuda-time-table.html

                  With the MLP head dominating the parameter count, the layers doing the
                  geometric product are a small share of the step, so the ratios are the
                  flattest on this page: every rotorch column sits at 2.3× to 2.6× cgenn at
                  16384, against the convex hull's spread of 2.0× to 6.6×. :code:`--compile
                  model` still has the lowest floor, 2.4 ms against cgenn's 7.1.

               .. tab-item:: Memory
                  :sync: memory

                  .. raw:: html
                     :file: _static/o5-cuda-memory.svg

                  .. raw:: html
                     :file: _static/o5-cuda-memory-table.html

                  Peak MiB of a forward and backward. Absolute numbers are far smaller than the
                  convex hull's, 1.2 GiB against 11.6 at batch 16384, since a plain MLP head
                  does not carry the 32-blade multivectors of a product layer, but the ratio is
                  in the same range: rotorch uses between an eighth and a fifth of cgenn's peak.

               .. tab-item:: Parameters
                  :sync: parameters

                  ==============  ==========  ========
                  implementation  parameters  val loss
                  ==============  ==========  ========
                  cgenn           344,077     0.0037
                  rotorch         343,125     0.0057
                  ==============  ==========  ========

                  At batch 16384. The two implementations differ by 952 parameters out of a
                  third of a million, since only the product layers before the head carry a
                  sparsity difference at all.

               .. tab-item:: Start-up
                  :sync: startup

                  ====================  =======  =======
                  run                   fastest  slowest
                  ====================  =======  =======
                  cgenn                 0.3      0.3
                  cgenn-compiled        2.3      3.1
                  rotorch               0.3      0.3
                  rotorch-operators     2.4      3.5
                  rotorch-model         2.1      15.4
                  rotorch-triton        7.6      22.0
                  ====================  =======  =======

                  Seconds to the first step. The smallest compile times on this page: this
                  model has fewer distinct product-layer shapes for inductor or the triton
                  autotuner to specialize than either the convex hull or O(3).

         .. tab-item:: CPU
            :sync: cpu

            .. tab-set::

               .. tab-item:: Time
                  :sync: time

                  .. raw:: html
                     :file: _static/o5-cpu-time.svg

                  .. raw:: html
                     :file: _static/o5-cpu-time-table.html

                  Eager rotorch is ahead of cgenn at every batch size here too, as on O(3), and
                  :code:`--compile model` reaches 4.0× at batch 2048 against the 8.4× and 14.4×
                  that the smaller and sparser O(3) and convex hull models reach: the MLP head
                  is the same cost for both implementations, so there is less of the step left
                  for compiling to improve.

               .. tab-item:: Memory
                  :sync: memory

                  Not measured on the cpu; see the GPU tab for the ratios.

               .. tab-item:: Parameters
                  :sync: parameters

                  ==============  ==========  ========
                  implementation  parameters  val loss
                  ==============  ==========  ========
                  cgenn           344,077     0.0046
                  rotorch         343,125     0.0061
                  ==============  ==========  ========

                  At batch 2048.

               .. tab-item:: Start-up
                  :sync: startup

                  =================  =======  =======
                  run                fastest  slowest
                  =================  =======  =======
                  cgenn              0.1      0.1
                  cgenn-compiled     3.7      4.8
                  rotorch            0.1      0.2
                  rotorch-operators  3.6      4.5
                  rotorch-model      3.3      3.5
                  =================  =======  =======

Why the sparsity pays
---------------------

Only 1024 of the 32768 entries of the :math:`Cl(5)` Cayley tensor are nonzero, so cgenn's
einsum performs 32 times the necessary multiply-adds. Nothing on this page is 32× faster, for
two reasons, and neither is about how much data fits in cache.

The first is that the product is not the whole layer. Per sample, each of the two linear maps
in a product layer costs 32 blades × 32 in × 32 out = 32,768 multiply-adds, and the sparse
product costs 1024 paths × 32 features = the same 32,768 again. The dense product costs 32
times that, 1,048,576. So the layer is 1.11 M multiply-adds for cgenn against 98 K for rotorch:
a factor of 11, not 32, because two thirds of rotorch's arithmetic is work both implementations
do identically.

The second is where the intermediates live. The geometric product of two full multivectors,
batch 512 and 32 features, counting only the multiply-adds that are not multiplications by a
structural zero:

======================  =====  ==============
run                        ms  useful GFLOP/s
======================  =====  ==============
sparse, eager            7.50             4.5
sparse, compiled         1.18            28.5
dense einsum            16.20             2.1
dense einsum, compiled   6.57             5.1
======================  =====  ==============

Eager, every one of those thousand multiply-adds is a separate torch call that reads two arrays
and writes a third, so each useful operation drags about a dozen bytes through memory and the
sparse product reaches 4.5 GFLOP/s. The dense einsum loads each value once and keeps its
intermediates in registers, so it runs at 67 GFLOP/s of raw arithmetic while doing 32 times too
much of it, and still beats the sparse product by a factor of two. Fusing the sparse product is
what removes that traffic: compiled it reaches 28.5 GFLOP/s and is 5.6× faster than the
compiled dense one. This is the whole story of the eager rotorch column, and of why triton and
:code:`--compile model` are the two that win.

The same reasoning applies to rotorch's own layers, which is why the ones that hold a parameter
per grade contract over the blade axis in one go rather than a blade at a time: 32 small
einsums cost 0.61 ms where a single batched one costs 0.19, 55 against 178 GFLOP/s. Since the
coefficients of a multivector are one tensor, the layer only has to gather the parameter of
each blade first, and ``einops`` contracts multivectors directly, so this costs nothing in
readability.

Agreement with cgenn
--------------------

The two implementations agree to machine precision, module by module, in :math:`Cl(2)`,
:math:`Cl(3)`, :math:`Cl(3,1)` and :math:`Cl(3,0,1)`.

lorentz is checked whole rather than module by module, since its layers are wired together in a
way the others are not. Copying cgenn's weights into rotorch's model -- grade by grade for the
linear maps, path by path for the products, and column by column for the plain layers that read
invariants, since a grade cgenn allocates for and rotorch does not have contributes nothing --
and running both over the same jets leaves at most :math:`5 \cdot 10^{-13}` between their
logits in double precision, four rounds of message passing deep. The same check in single
precision leaves 2%, which is not disagreement but cancellation: a momentum of a few hundred
GeV squares to a few hundred thousand, and every invariant in this model is a difference of
such numbers.

The validation losses in the tabs above differ between the two implementations because they do
not start from the same place: their parameter counts differ, so the initial weights are drawn
differently, and since building the model draws from the same generator the batches are
shuffled differently too. Over five seeds of the o3 example they overlap, at 0.0105 to 0.0223
for rotorch against 0.0147 to 0.0339 for cgenn, and by 512 steps both settle around 0.001. The
gap between any two runs is the seed, not the implementation.

Other devices
-------------

The examples also run on :code:`--device mps`, where kernel launch overhead dominates below a
batch size of a few hundred and the GPU wins above it. Neither :code:`--compile` mode runs
there, since inductor's Metal backend cannot compile these kernels: the wide ones exceed
Metal's limit of about 31 buffer arguments per kernel, one per blade, and the rest fail to
build their shaders. :code:`--backend triton` is cuda only, and falls back to the torch backend
anywhere else.
