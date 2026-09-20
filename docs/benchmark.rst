Benchmark
=========

The convex hull regression of `Clifford Group Equivariant Neural Networks
<https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks>`_ (cgenn), run
with rotorch's layers and with the original implementation. The task is to predict the volume
of the convex hull of 16 points in :math:`\mathbb{R}^5`, so the algebra is :math:`Cl(5)`
and the model is a linear embedding followed by four geometric product layers.

Both implementations are driven by the same script, :code:`examples/hulls.py`, so they see
the same data, the same schedule and the same optimizer::

    python examples/hulls.py                       # rotorch
    python examples/hulls.py --compile operators   # torch.compile as kingdon's operator wrapper
    python examples/hulls.py --compile model       # torch.compile over the whole model
    python examples/hulls.py --impl cgenn          # the original implementation

Timings are the median over 64 steps of forward, backward and optimizer step, after eight
warm-up steps, and then the faster of two such runs, on an idle Apple M2 with torch 2.14 and
python 3.12. The same matrix on a workstation, cpu and gpu side by side and weighed as well as
timed, is under :ref:`workstation` below.

Throughput
----------

Milliseconds per step, and the speedup over cgenn:

.. list-table::
   :header-rows: 1
   :widths: 8 14 16 16 18 16

   * - batch
     - cgenn
     - cgenn, compiled
     - rotorch
     - rotorch, operators
     - rotorch, model
   * - 32
     - 39.9
     - 31.6 (1.3×)
     - 28.8 (1.4×)
     - 10.8 (3.7×)
     - 9.6 (4.2×)
   * - 128
     - 101.1
     - 64.9 (1.6×)
     - 39.7 (2.5×)
     - 25.7 (3.9×)
     - 24.6 (4.1×)
   * - 512
     - 359.6
     - 205.3 (1.8×)
     - 119.4 (3.0×)
     - 68.6 (5.2×)
     - 79.9 (4.5×)
   * - 2048
     - 1475.4
     - 932.7 (1.6×)
     - 479.6 (3.1×)
     - 315.6 (4.7×)
     - 345.9 (4.3×)

Uncompiled, the lead grows with the batch size because cgenn contracts against the dense
Cayley tensor, paying for all :math:`32^3` entries whatever the input grades are, while rotorch
only evaluates the paths that the grades present can actually reach. Compiled, rotorch is between
four and five and a half times faster than cgenn.

Compiling
---------

:code:`--compile operators` hands :func:`torch.compile` to kingdon as its wrapper, so every
operator kingdon generates for the algebra is compiled on its own: 98 functions for this
model, plus their backward passes. :code:`--compile model` compiles the training step
instead, which fuses across those operators rather than stopping at each one.

Whole model compilation works because a multivector is a pytree whose coefficients are one
tensor and whose keys are static context, and because nothing in the path raises: the model
traces to **one graph with no breaks**, and :code:`fullgraph=True` succeeds and reproduces
the the eager loss and gradients exactly. Two caveats:

* Python 3.12 or later. Before that, :code:`functools.cached_property` takes a lock, which
  :class:`~kingdon.multivector.MultiVector` uses for its shape and grades, and dynamo breaks
  the graph at a lock.
* Not :code:`dynamic=True`, which makes the blade keys symbolic ints and their
  :code:`bit_count` untraceable. With the default, the second batch size compiles dynamic and
  is the last compile; :func:`torch._dynamo.mark_dynamic` on the batch axis of the
  coefficients makes it one. :code:`tests/test_compile.py` checks both.

cgenn cannot be compiled that strictly, since it
indexes its weights with a boolean mask, whose result has a data dependent shape; its
:code:`--compile model` column is therefore compiled with graph breaks. The lorentz model below
traces to one graph as well, message passing, batch norms and gathers included.

Which of the two wins depends on the batch size, and they cross between 128 and 512.
Compiling the model removes almost all of the per step python and dispatch cost, which is
what dominates a small batch. Compiling each operator gives inductor smaller graphs to
schedule, which suits the memory bound work of a large batch better, and costs a fraction as
long to compile: 98 small graphs rather than one large one.

Other examples
--------------

The other four examples of cgenn are set up the same way, each with its own data and model but
the same loop, so they can be compared in the same terms. Milliseconds per step at each
example's own defaults, over 64 steps:

.. list-table::
   :header-rows: 1
   :widths: 10 20 10 10 12

   * - example
     - what it predicts
     - cgenn
     - rotorch
     - rotorch, model
   * - hulls
     - the volume of a convex hull in 5D
     - 39.6
     - 29.2
     - 9.5 (4.2×)
   * - o3
     - the determinant of three vectors in 3D
     - 6.3
     - 4.0
     - 3.4 (1.9×)
   * - o5
     - an O(5) invariant of two vectors in 5D
     - 3.7
     - 2.0
     - 1.2 (3.1×)
   * - nbody
     - where five charged particles end up
     - 258.4
     - 229.8
     - 94.7 (2.7×)
   * - lorentz
     - which jets came from a top quark
     - 329.7
     - 484.2
     - 194.0 (1.7×)

Each reaches the same validation loss as cgenn does on the same data, with the parameter
counts below. o3 gains least from compiling because its model is small enough that a step is
mostly fixed cost either way. The nbody and lorentz datasets are simulated by the examples rather
than read from the files the EGNN repository and the top tagging reference set ship, so their
trajectories and their jets are not the ones cgenn trains on.

lorentz is the one example where rotorch is slower than cgenn until it is compiled, and it is slower
for the reason given under `Where the time goes`_. Its products are wide, mixing 27 input features
into 8 output ones over the 2,912 edges of a batch, so every intermediate is an array of some six
hundred thousand numbers that eager mode writes out and reads back. Its multivectors are dense
besides: a vector in :math:`Cl(1,3)` has reached every grade by the end of the first layer, so
from the second on there is no sparsity left to spend, only the 16× fewer multiply-adds that the
sparse Cayley table saves. Fusing those intermediates is what turns 1.5× slower into 1.7× faster.

The validation losses of rotorch and cgenn differ even though they train on the same data, because
they do not start from the same place: their parameter counts differ, so the initial weights are
drawn differently, and since building the model draws from the same generator the batches are
shuffled differently too. Over five seeds of the o3 example they overlap, at 0.0105 to 0.0223
for rotorch against 0.0147 to 0.0339 for cgenn, and by 512 steps both settle around 0.001. The gap
between any two runs is the seed, not the implementation; the layers themselves agree to machine
precision when handed the same weights.

Where the time goes
-------------------

Only 1024 of the 32768 entries of the :math:`Cl(5)` Cayley tensor are nonzero, so cgenn's
einsum performs 32 times the necessary multiply-adds. The measured advantage is nowhere near
32 times, and fitting the timings above as a fixed cost plus a cost per sample says why
(:math:`r^2 \geq 0.997`):

==================  ===============  =================
run                 fixed per step   marginal
==================  ===============  =================
cgenn               7.8 ms           715 µs / sample
cgenn, compiled     1.5 ms           452 µs / sample
rotorch             12.6 ms          227 µs / sample
rotorch, operators  1.8 ms           152 µs / sample
rotorch, model      1.0 ms           168 µs / sample
==================  ===============  =================

The marginal cost, which is the part that scales with the data, is where the sparsity shows
up: 3.1× cheaper than cgenn uncompiled, 4.7× compiled. It is not 32×, for two reasons, and
neither is about how much data fits in cache.

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

Eager, every one of those thousand multiply-adds is a separate torch call that reads two
arrays and writes a third, so each useful operation drags about a dozen bytes through memory
and the sparse product reaches 4.5 GFLOP/s. The dense einsum loads each value once and keeps
its intermediates in registers, so it runs at 67 GFLOP/s of raw arithmetic while doing 32
times too much of it, and still beats the sparse product by a factor of two. Fusing the
sparse product is what removes that traffic: compiled it reaches 28.5 GFLOP/s and is 5.6×
faster than the compiled dense one.

The same reasoning applies to rotorch's own layers, which is why the ones that hold a parameter
per grade contract over the blade axis in one go rather than a blade at a time: 32 small
einsums cost 0.61 ms where a single batched one costs 0.19, 55 against 178 GFLOP/s. Since the
coefficients of a multivector are one tensor, the layer only has to gather the parameter of
each blade first, and ``einops`` contracts multivectors directly, so this costs nothing in
readability. It is mostly a fixed cost saving, one kernel where there were 32, which is why it
moved the fixed cost of the fit above by a factor of two and left the marginal cost alone.

The fixed cost is the mirror image. rotorch pays 12.6 ms per step before any data is touched:
one forward and backward calls into 98 generated operators, each with its own dictionary
lookups, multivector construction and dispatch. That is python and framework time, not
arithmetic, and at batch 32 it is still 44% of the step. Compiling removes nearly all of it,
12.6 ms down to 1, which is the whole of the 4.2× at batch 32.

Parameters
----------

=========  ==========  ==========
example         cgenn     rotorch
=========  ==========  ==========
hulls          58,849      38,881
o3              8,657       4,973
o5            344,077     343,125
nbody         134,625     126,645
lorentz       320,594     303,525
=========  ==========  ==========

Same function, fewer parameters, a third of them in the case of hulls. Its input is a pure
vector, so the grades only fill in as the products generate them: the first product layer sees
grades (0, 1) and needs 5 path weights, the second (0, 1, 2) and 14, the third 45, the fourth
56. cgenn allocates all 56 paths and all six subspace matrices in every layer; the unreachable
ones multiply zero coefficients, contributing nothing and receiving no gradient. o5 is the
exception at almost the same count, since nearly all of its parameters sit in a plain MLP head
that both implementations share.

The two implementations agree to machine precision, module by module, in :math:`Cl(2)`,
:math:`Cl(3)`, :math:`Cl(3,1)` and :math:`Cl(3,0,1)`, and every column above reaches the same
validation loss, 21.7 to 22.7 depending on the batch size.

lorentz is checked whole rather than module by module, since its layers are wired together in a
way the others are not. Copying cgenn's weights into rotorch's model -- grade by grade for the
linear maps, path by path for the products, and column by column for the plain layers that read
invariants, since a grade cgenn allocates for and rotorch does not have contributes nothing -- and
running both over the same jets leaves at most :math:`5 \cdot 10^{-13}` between their logits in
double precision, four rounds of message passing deep. The same check in single precision leaves
2%, which is not disagreement but cancellation: a momentum of a few hundred GeV squares to a few
hundred thousand, and every invariant in this model is a difference of such numbers.

Start-up
--------

Compilation is paid on the first step:

=======================  ===================
run                      first step
=======================  ===================
rotorch                  0.2 to 0.8 s
compiled, warm cache     6 to 8 s
compiled, cold cache     139 s to 276 s
=======================  ===================

The cold range is the two modes: 139 s to compile the operators one at a time, 276 s for the
model as one graph. Inductor caches its kernels under :code:`$TMPDIR/torchinductor_$USER`,
some 60 MB for this model, so that is paid once per machine rather than once per run.
Compilation is specialized per shape, so each batch size compiles anew, as does the ragged
final batch of a validation loader. At batch 32, :code:`--compile model` breaks even against
the uncompiled run after about 14,000 steps cold, or 370 warm.

Compiling the model needs kingdon's operator cache to be warm, since kingdon generates its
operators on the first call and dynamo cannot trace code generation. The example therefore
runs one step eagerly before compiling.

.. _workstation:

On a workstation
----------------

The whole matrix again on one machine that has a card worth using: an **NVIDIA RTX A4000**
(16 GB) beside an **AMD Zen 3** (family 25), Windows 11, torch 2.9.1 with cuda 12.8, triton 3.5
and python 3.12. Same script, same protocol as above -- the median of 64 steps after eight
warm-up ones, the faster of two runs -- driven by :code:`examples/sweep.py`, which also weighs
each run: the peak allocation of a forward, and of a forward and backward together, the way
`flash-clifford <https://github.com/tBuLi/flash-kingdon-clifford>`_ reports its memory. The
batch size goes to 16384 on the card, where 2048 was as far as the cpu was worth taking.

These are a different machine from the numbers at the top of this page, so read each device
against itself rather than against the M2.

.. tab-set::

   .. tab-item:: CUDA
      :sync: cuda

      .. raw:: html
         :file: _static/hulls-cuda-time.svg

      Milliseconds per step, and the speedup over cgenn:

      .. list-table::
         :header-rows: 1
         :widths: 10 14 18 16 20

         * - batch
           - cgenn
           - cgenn, compiled
           - rotorch
           - rotorch, operators
         * - 32
           - 31.1
           - 25.4 (1.22×)
           - 224.4 (0.14×)
           - 33.5 (0.93×)
         * - 128
           - 30.8
           - 24.7 (1.25×)
           - 230.2 (0.13×)
           - 33.6 (0.92×)
         * - 512
           - 30.6
           - 26.3 (1.16×)
           - 237.1 (0.13×)
           - 38.0 (0.81×)
         * - 2048
           - 69.9
           - 62.0 (1.13×)
           - 239.0 (0.29×)
           - 43.0 (1.63×)
         * - 4096
           - 128.0
           - 113.8 (1.12×)
           - 240.1 (0.53×)
           - 58.4 (2.19×)
         * - 8192
           - 248.0
           - 219.9 (1.13×)
           - 268.6 (0.92×)
           - 109.3 (2.27×)
         * - 16384
           - 490.5
           - 443.5 (1.11×)
           - 346.6 (1.42×)
           - 202.2 (2.43×)

      Every column is flat to batch 512 and rises after it, which is the same shape the cpu
      table has; what differs is the height of the flat part and the slope after it. Three of
      the four sit on a floor between 25 and 38 ms, near enough the same, because a step that small
      is fixed cost whoever runs it. Eager rotorch's floor is 224 ms, seven times higher, and
      it holds that floor all the way to 4096: a step that does not notice a hundred and
      twenty-eight times the data, because it is not doing arithmetic, it is waiting for
      python to launch its next kernel. Past 2048 the marginal cost takes over and sets the
      order, and :code:`--compile operators` is the fastest column on the card from there on.

      Fitting ms/step as a fixed cost plus a cost per sample, over all seven batch sizes:

      ==================  ===============  =================  =======
      run                 fixed per step   marginal           r²
      ==================  ===============  =================  =======
      cgenn               19.5 ms          28.4 µs / sample   0.998
      cgenn, compiled     15.1 ms          25.8 µs / sample   0.998
      rotorch             224.0 ms         6.9 µs / sample    0.954
      rotorch, operators  27.6 ms          10.4 µs / sample   0.988
      ==================  ===============  =================  =======

      The marginal costs are the sparsity: 10.4 µs per sample against cgenn's 28.4, 2.7×
      cheaper, and that is the ratio the speedup column is climbing towards as the fixed cost
      stops mattering -- 1.6× at 2048, 2.2× at 4096, 2.4× at 16384. Eager rotorch's 6.9 µs is
      the lowest marginal cost of the four and buys nothing, because 224 ms of launches is in
      front of it.

      .. raw:: html
         :file: _static/hulls-cuda-memory.svg

      Peak MiB of a forward and backward, and the ratio to cgenn, where below one is a saving:

      .. list-table::
         :header-rows: 1
         :widths: 10 14 18 16 20

         * - batch
           - cgenn
           - cgenn, compiled
           - rotorch
           - rotorch, operators
         * - 32
           - 58.6
           - 58.3 (0.99×)
           - 23.4 (0.40×)
           - 21.5 (0.37×)
         * - 128
           - 128.3
           - 127.2 (0.99×)
           - 38.1 (0.30×)
           - 30.5 (0.24×)
         * - 512
           - 407.0
           - 401.5 (0.99×)
           - 97.0 (0.24×)
           - 66.6 (0.16×)
         * - 2048
           - 1524.8
           - 1503.4 (0.99×)
           - 334.4 (0.22×)
           - 213.2 (0.14×)
         * - 4096
           - 3010.2
           - 2970.0 (0.99×)
           - 646.6 (0.21×)
           - 404.6 (0.13×)
         * - 8192
           - 5982.2
           - 5901.2 (0.99×)
           - 1275.1 (0.21×)
           - 791.1 (0.13×)
         * - 16384
           - 11929.3
           - 11755.6 (0.99×)
           - 2518.7 (0.21×)
           - 1550.6 (0.13×)

      The memory is the cleaner result of the two, because nothing about it is a matter of
      launch overhead: it is the same 32 blades of every intermediate that the dense einsum
      writes down and the sparse product does not. rotorch holds a fifth of cgenn's memory
      eager and an eighth compiled, and the ratio is settled by batch 512 and flat from there.
      At 16384 that is 1.5 GiB against 11.9 GiB: cgenn is within four gigabytes of filling the
      card and cannot have the next doubling at all, while rotorch is still using under a
      tenth of it.

      Compiling barely moves it either way: inductor fuses arithmetic, not activations, and an
      activation that the backward will want has to exist whoever wrote it. A forward on its
      own is within a percent of the figures above for the three uncompiled columns -- these
      models keep almost everything for the backward -- and 13% under them for
      :code:`--compile operators`, which is the only column where fusion drops an intermediate
      the backward turns out not to need.

      :code:`--compile model` has no column: it failed on this machine too, in seven seconds,
      before compiling anything. The log stayed on that machine, so the reason is not in the
      csv, but the timing matches the failure of the three compiled cpu runs below to the
      second, and those are known to be a missing C++ compiler.

   .. tab-item:: CPU
      :sync: cpu

      .. raw:: html
         :file: _static/hulls-cpu-time.svg

      Milliseconds per step, and the speedup over cgenn:

      .. list-table::
         :header-rows: 1
         :widths: 10 16 20

         * - batch
           - cgenn
           - rotorch
         * - 32
           - 56.7
           - 54.2 (1.05×)
         * - 128
           - 116.0
           - 70.9 (1.64×)
         * - 512
           - 376.8
           - 145.0 (2.60×)
         * - 2048
           - 1137.1
           - 332.0 (3.42×)

      ==================  ===============  ==================  =======
      run                 fixed per step   marginal            r²
      ==================  ===============  ==================  =======
      cgenn               61.1 ms          530.2 µs / sample   0.996
      rotorch             58.4 ms          135.5 µs / sample   0.992
      ==================  ===============  ==================  =======

      The two start level, because at batch 32 a step is fixed cost for both, and separate by
      the marginal cost from there: 135.5 µs per sample against 530.2, 3.9× cheaper, which is
      the same story the M2 tells at the top of this page and close to the same number. There
      is no launch overhead on a cpu to hide it, which is why the cpu column needs no
      compiling to show the sparsity and the cuda column does.

      The three compiled configurations have no rows. Inductor writes C++ for the cpu and
      needs a compiler for it, and :code:`cl.exe` was not on the path of the shell that ran
      the sweep, so all three failed in about six seconds and the sweep recorded them and
      moved on. They are the one gap in this matrix; a run from a developer prompt would fill
      them in.

Compiling costs more on the card than it does on the M2 at the small batch sizes and less at
the large ones, and the cache matters more than the batch size does:

=======================  ==================  ==================
run                      first step, cold    first step, warm
=======================  ==================  ==================
cgenn                    0.3 to 1.0 s        --
rotorch                  0.9 to 1.1 s        --
cgenn, compiled          18 to 24 s          2.4 to 2.9 s
rotorch, operators       12 to 397 s         10 to 14 s
=======================  ==================  ==================

The 98 operators take twelve to twenty-six seconds to compile at batch 32 and 128 and about six
and a half minutes from 512 up, where the shapes are large enough that inductor stops taking the
cheap path. Warm, any of them is back in under fifteen seconds. At batch 16384, where compiling
saves 144 ms a step, that cold compile has paid for itself after about 2,700 steps, or 100 warm.

Devices
-------

Besides cuda, the example runs on :code:`--device mps`, where kernel launch overhead dominates
below a batch size of a few hundred and the GPU wins above it: at batch 32 rotorch takes 84.7
ms/step on mps against 28.8 on cpu, and at batch 2048 218.7 against 479.6. Neither :code:`--compile` mode
runs there, since inductor's Metal backend cannot compile these kernels: the wide ones exceed
Metal's limit of about 31 buffer arguments per kernel, one per blade, and the rest fail to
build their shaders.
