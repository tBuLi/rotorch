Benchmark
=========

This page lists the benchmarks of `rotorch` against `Clifford Group Equivariant Neural Networks
<https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks>`_ (cgenn).
All the benchmarks were done on a machine with the following specs: **NVIDIA RTX A4000**
(16 GB), AMD Ryzen Threadripper PRO 7955WX, Windows 11, torch 2.9.1 with cuda 13.0, triton 3.5
and python 3.14.

A note on the different scenarios that were benchmarked:

- `cgenn`: the cgenn implementation, without any optimization.
- `cgenn-model`: the entire cgenn implementation compiled with :func:`torch.compile`. This has graph breaks, since it
  indexes its weights with a boolean mask, whose result has a data dependent shape.
- `rotorch`: the rotorch example, without any optimization.
- `rotorch-operators`: hands :func:`torch.compile` to kingdon as its wrapper, so every
  operator kingdon generates for the algebra is compiled on its own.
- `rotorch-operators-model`: in addition to compiling the operators like `rotorch-operators`, this compiles the
  training step as a whole, fusing across those operators rather than stopping at each one. This results in
  **one graph with no breaks**, and :code:`fullgraph=True` succeeds.
- `rotorch-triton`: instead of applying :func:`torch.compile` to the kingdon optimized code, this hands kingdon a
  code printer that directly produces a triton kernel, to which :func:`triton.jit` is applied.
- `rotorch-triton-model`: in addition to triton kernels like `rotorch-triton`, this compiles the whole model.

The triton scenarios are only available on the GPU, whereas all the others exist for both CPU and GPU.


.. tab-set::

   .. tab-item:: Convex Hull

      The convex hull regression of `Clifford Group Equivariant Neural Networks
      <https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks>`_ (cgenn).
      The task is to predict the volume of the convex hull of 16 points in :math:`\mathbb{R}^5`,
      so the algebra is :math:`Cl(5)` and the model is a linear embedding followed by four geometric product layers.

      Both implementations are driven by the same script, :code:`examples/hulls.py`, so they see
      the same data, the same schedule and the same optimizer::

          python examples/hulls.py                       # rotorch
          python examples/hulls.py --compile operators   # torch.compile as kingdon's operator wrapper
          python examples/hulls.py --compile model       # torch.compile over the whole model
          python examples/hulls.py --backend triton      # one triton kernel per operator, cuda only
          python examples/hulls.py --impl cgenn          # the original implementation


      .. tab-set::

         .. tab-item:: GPU

            Measured on an **NVIDIA RTX A4000**.

            .. tab-set::

               .. tab-item:: Time

                  (Smaller is better.)

                  .. raw:: html
                     :file: _static/hulls-cuda-time.svg

                  Milliseconds per step, and the speedup over cgenn:

                  .. list-table::
                     :header-rows: 1
                     :widths: 8 12 16 14 18 16

                     * - batch
                       - cgenn
                       - cgenn, compiled
                       - rotorch
                       - rotorch, operators
                       - rotorch, triton
                     * - 32
                       - 24.1
                       - 24.9 (0.97×)
                       - 227.9 (0.11×)
                       - 34.0 (0.71×)
                       - 25.2 (0.96×)
                     * - 128
                       - 24.3
                       - 25.0 (0.97×)
                       - 228.7 (0.11×)
                       - 33.9 (0.72×)
                       - 25.2 (0.96×)
                     * - 512
                       - 26.8
                       - 26.6 (1.01×)
                       - 227.6 (0.12×)
                       - 37.9 (0.71×)
                       - 26.3 (1.02×)
                     * - 2048
                       - 68.3
                       - 62.0 (1.10×)
                       - 225.6 (0.30×)
                       - 42.8 (1.60×)
                       - 29.3 (2.33×)
                     * - 4096
                       - 129.1
                       - 114.0 (1.13×)
                       - 242.0 (0.53×)
                       - 58.1 (2.22×)
                       - 35.4 (3.65×)
                     * - 8192
                       - 247.7
                       - 220.8 (1.12×)
                       - 267.3 (0.93×)
                       - 109.0 (2.27×)
                       - 48.9 (5.07×)
                     * - 16384
                       - 489.1
                       - 445.5 (1.10×)
                       - 335.4 (1.46×)
                       - 202.4 (2.42×)
                       - 81.5 (6.00×)

               .. tab-item:: Memory

                  (Smaller is better.)

                  .. raw:: html
                     :file: _static/hulls-cuda-memory.svg

               .. tab-item:: Parameters

                    =========  ==========  ==========
                    example         cgenn     rotorch
                    =========  ==========  ==========
                    hulls          58,849      38,881
                    =========  ==========  ==========

               .. tab-item:: Start-up

                    =======================  ===================
                    run                      first step
                    =======================  ===================
                    rotorch                  0.2 to 0.8 s
                    compiled, warm cache     6 to 8 s
                    compiled, cold cache     139 s to 276 s
                    =======================  ===================

         .. tab-item:: CPU

            Measured on an **AMD Ryzen Threadripper PRO 7955WX**.

   .. tab-item:: Lorentz

      Coming soon.

   .. tab-item:: O(3)

      .. tab-set::

         .. tab-item:: GPU

            Measured on an **NVIDIA RTX A4000**.

            .. tab-set::

               .. tab-item:: Time

                  (Smaller is better.)

                  .. raw:: html
                     :file: _static/o3-cuda-time.svg

               .. tab-item:: Memory

                  (Smaller is better.)

                  .. raw:: html
                     :file: _static/o3-cuda-memory.svg

         .. tab-item:: CPU

            Measured on an **AMD Ryzen Threadripper PRO 7955WX**.

            (Smaller is better.)

            .. raw:: html
               :file: _static/o3-cpu-time.svg




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

This machine has a column the M2 cannot run: :code:`--backend triton` asks kingdon for one
triton kernel per operator instead of one torch call per symbolic multiply. It is a codegen
switch rather than a compiler pass, emitted from the same polynomials the torch calls come
from, and :func:`torch.compile` never sees the model. Every rotorch column on the card reaches
the validation loss of the eager one to four decimals, triton and compiled operators alike, so
the kernels and the fusions agree with what they replace.

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
         :widths: 8 12 16 14 18 16

         * - batch
           - cgenn
           - cgenn, compiled
           - rotorch
           - rotorch, operators
           - rotorch, triton
         * - 32
           - 24.1
           - 24.9 (0.97×)
           - 227.9 (0.11×)
           - 34.0 (0.71×)
           - 25.2 (0.96×)
         * - 128
           - 24.3
           - 25.0 (0.97×)
           - 228.7 (0.11×)
           - 33.9 (0.72×)
           - 25.2 (0.96×)
         * - 512
           - 26.8
           - 26.6 (1.01×)
           - 227.6 (0.12×)
           - 37.9 (0.71×)
           - 26.3 (1.02×)
         * - 2048
           - 68.3
           - 62.0 (1.10×)
           - 225.6 (0.30×)
           - 42.8 (1.60×)
           - 29.3 (2.33×)
         * - 4096
           - 129.1
           - 114.0 (1.13×)
           - 242.0 (0.53×)
           - 58.1 (2.22×)
           - 35.4 (3.65×)
         * - 8192
           - 247.7
           - 220.8 (1.12×)
           - 267.3 (0.93×)
           - 109.0 (2.27×)
           - 48.9 (5.07×)
         * - 16384
           - 489.1
           - 445.5 (1.10×)
           - 335.4 (1.46×)
           - 202.4 (2.42×)
           - 81.5 (6.00×)

      Every column is flat to batch 512 and rises after it, which is the same shape the cpu
      table has; what differs is the height of the flat part and the slope after it. Four of
      the five sit on a floor between 24 and 34 ms, near enough the same, because a step that small
      is fixed cost whoever runs it. Eager rotorch's floor is 228 ms, nine times higher, and
      it holds that floor all the way to 4096: a step that does not notice a hundred and
      twenty-eight times the data, because it is not doing arithmetic, it is waiting for
      python to launch its next kernel. Past 2048 the marginal cost takes over and sets the
      order, and :code:`--backend triton` is the fastest column on the card from there on: 6.0×
      cgenn at 16384, where :code:`--compile operators` reaches 2.4×.

      Fitting ms/step as a fixed cost plus a cost per sample, over all seven batch sizes:

      ==================  ===============  =================  =======
      run                 fixed per step   marginal           r²
      ==================  ===============  =================  =======
      cgenn               15.4 ms          28.7 µs / sample   0.999
      cgenn, compiled     15.0 ms          25.9 µs / sample   0.998
      rotorch             221.1 ms         6.6 µs / sample    0.967
      rotorch, operators  27.6 ms          10.4 µs / sample   0.987
      rotorch, triton     23.5 ms          3.4 µs / sample    0.991
      ==================  ===============  =================  =======

      The marginal costs are the sparsity, and triton's 3.4 µs per sample is the lowest of the
      five: 8.4× cheaper than cgenn's 28.7, where compiled operators are 2.7× cheaper. That is
      the ratio each speedup climbs towards as its fixed cost stops mattering -- triton is 2.3×
      at 2048, 3.7× at 4096, 6.0× at 16384, and has not levelled off yet. Eager rotorch's 6.6 µs
      buys nothing, because 221 ms of launches is in front of it, and the two rotorch backends
      do the same multiply-adds: one kernel keeps the intermediates in registers where a
      thousand torch calls write each of them out and read it back, which is both why triton's
      fixed cost is a tenth of eager's and why its marginal cost is half.

      .. raw:: html
         :file: _static/hulls-cuda-memory.svg

      Peak MiB of a forward and backward, and the ratio to cgenn, where below one is a saving:

      .. list-table::
         :header-rows: 1
         :widths: 8 12 16 14 18 16

         * - batch
           - cgenn
           - cgenn, compiled
           - rotorch
           - rotorch, operators
           - rotorch, triton
         * - 32
           - 58.6
           - 58.3 (0.99×)
           - 23.4 (0.40×)
           - 21.5 (0.37×)
           - 22.6 (0.39×)
         * - 128
           - 128.3
           - 127.2 (0.99×)
           - 38.1 (0.30×)
           - 30.5 (0.24×)
           - 34.9 (0.27×)
         * - 512
           - 407.0
           - 401.5 (0.99×)
           - 97.0 (0.24×)
           - 66.6 (0.16×)
           - 84.2 (0.21×)
         * - 2048
           - 1524.8
           - 1503.4 (0.99×)
           - 334.4 (0.22×)
           - 213.2 (0.14×)
           - 308.9 (0.20×)
         * - 4096
           - 3010.2
           - 2970.0 (0.99×)
           - 646.6 (0.21×)
           - 404.6 (0.13×)
           - 579.4 (0.19×)
         * - 8192
           - 5982.2
           - 5901.2 (0.99×)
           - 1275.1 (0.21×)
           - 791.1 (0.13×)
           - 1108.8 (0.19×)
         * - 16384
           - 11929.3
           - 11755.6 (0.99×)
           - 2518.7 (0.21×)
           - 1550.7 (0.13×)
           - 2147.2 (0.18×)

      The memory is the cleaner result of the two, because nothing about it is a matter of
      launch overhead: it is the same 32 blades of every intermediate that the dense einsum
      writes down and the sparse product does not. rotorch holds a fifth of cgenn's memory
      eager, a little under a fifth through triton and an eighth with compiled operators, and
      every ratio is settled by batch 512 and flat from there. At 16384 that is 2.5, 2.1 and
      1.5 GiB against 11.7: cgenn is within four and a half gigabytes of filling the card and
      cannot have the next doubling at all, while the three rotorch columns are using between a
      tenth and a sixth of it.

      Compiling barely moves it either way: inductor fuses arithmetic, not activations, and an
      activation that the backward will want has to exist whoever wrote it. A forward on its
      own is within a percent of the figures above for cgenn, its compiled form and eager
      rotorch -- these models keep almost everything for the backward -- and up to 13% under
      them for triton and compiled operators, the two columns where a fused kernel drops an
      intermediate the backward turns out not to need.

      :code:`--compile model` has no column on either backend. Both failed at batch 32, the
      torch one after 7 s and the triton one after 41 s, and the sweep classified neither as
      the compiler failure that the four compiled cpu runs below are. The logs stayed on that
      machine, so the reason is not in the csv.

   .. tab-item:: CPU
      :sync: cpu

      .. raw:: html
         :file: _static/hulls-cpu-time.svg

      Milliseconds per step, and the speedup over cgenn:

      .. list-table::
         :header-rows: 1
         :widths: 10 14 18 18

         * - batch
           - cgenn
           - rotorch
           - rotorch, triton
         * - 32
           - 62.3
           - 75.6 (0.82×)
           - 76.7 (0.81×)
         * - 128
           - 128.0
           - 93.7 (1.37×)
           - 93.3 (1.37×)
         * - 512
           - 378.5
           - 143.6 (2.64×)
           - 142.5 (2.66×)
         * - 2048
           - 1079.2
           - 323.7 (3.33×)
           - 325.4 (3.32×)

      ==================  ===============  ==================  =======
      run                 fixed per step   marginal            r²
      ==================  ===============  ==================  =======
      cgenn               74.7 ms          496.0 µs / sample   0.995
      rotorch             76.8 ms          121.2 µs / sample   0.999
      rotorch, triton     76.6 ms          121.9 µs / sample   0.999
      ==================  ===============  ==================  =======

      rotorch is a fifth behind at batch 32, where a step is mostly fixed cost for both, and
      then the marginal cost takes over: 121.2 µs per sample against 496.0, 4.1× cheaper,
      which is the same story the M2 tells at the top of this page. There is no launch overhead on a cpu to hide it, which is why the cpu column
      needs no compiling to show the sparsity and the cuda column does.

      :code:`--backend triton` has a row here and no line on the figure. There is no triton for
      the cpu, so kingdon's dispatch falls back to the torch backend as soon as it sees that no
      argument is a cuda tensor, and this is the same column twice: never more than 1.5% apart
      at any batch size, on identical validation losses. Only the first step is longer, 0.8 to 1.3 s
      against 0.5 to 0.9, which is what the path that goes unused costs to set up.

      The four compiled configurations have no rows. Inductor writes C++ for the cpu and
      needs a compiler for it, and :code:`cl.exe` was not on the path of the shell that ran
      the sweep, so all four failed in six to eighteen seconds and the sweep recorded them and
      moved on. They are the one gap in this matrix; a run from a developer prompt would fill
      them in.

Start-up is paid per process, and on the card the ranges say more about what was already in a
cache than about the batch size:

=======================  ===================  ===================
run                      first step, rep 1    first step, rep 2
=======================  ===================  ===================
cgenn                    0.2 to 1.0 s         0.2 to 1.0 s
rotorch                  0.9 to 1.0 s         0.9 to 1.1 s
cgenn, compiled          2.7 to 3.2 s         2.0 to 2.5 s
rotorch, operators       14.4 to 15.5 s       9.9 to 11.5 s
rotorch, triton          36 to 204 s          35 to 39 s
=======================  ===================  ===================

Inductor's kernels were already under :code:`$TMPDIR/torchinductor_$USER` from an earlier sweep
on this machine, so neither compiled column above is cold and the gap between the two reps is
process warm-up rather than compilation. In that earlier sweep, with the cache empty,
:code:`--compile operators` took twelve to twenty-six seconds at batch 32 and 128 and about six
and a half minutes from 512 up, where the shapes are large enough that inductor stops taking
the cheap path.

Triton does not warm up the same way: 35 to 39 s in every rep at every batch size, because the
tile search compiles a candidate per operator and the autotuner then times the survivors, once
per process. Only the very first process paid more than that, 204 s at batch 32 and 58 s at 128.
At batch 16384 it is still cheap: triton saves 254 ms a step against eager rotorch and has paid
for itself after about 140 steps, where :code:`--compile operators` saves 133 ms and pays back
after 70. At 512 and below neither saves anything against cgenn, so there the start-up buys
parity and nothing more.

Devices
-------

Besides cuda, the example runs on :code:`--device mps`, where kernel launch overhead dominates
below a batch size of a few hundred and the GPU wins above it: at batch 32 rotorch takes 84.7
ms/step on mps against 28.8 on cpu, and at batch 2048 218.7 against 479.6. Neither :code:`--compile` mode
runs there, since inductor's Metal backend cannot compile these kernels: the wide ones exceed
Metal's limit of about 31 buffer arguments per kernel, one per blade, and the rest fail to
build their shaders. :code:`--backend triton` is cuda only, and falls back to the torch backend
anywhere else.
