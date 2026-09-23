API reference
=============

Both :mod:`rotorch.nn` and :mod:`rotorch.models` are organised one subpackage per architecture.
cgenn and GATr are implemented so far, and others will appear alongside them rather than in
place of them. What the two share, being about multivectors rather than about either paper,
lives in :mod:`rotorch.nn.utils`.

Layers
------

.. automodule:: rotorch.nn

cgenn
^^^^^

.. The layers are defined in submodules and re-exported from ``rotorch.nn.cgenn``, so
   ``imported-members`` is what documents them under the path users actually import.

.. automodule:: rotorch.nn.cgenn
   :members:
   :imported-members:
   :show-inheritance:

GATr
^^^^

.. automodule:: rotorch.nn.gatr
   :members:
   :imported-members:
   :show-inheritance:

Models
------

.. automodule:: rotorch.models

cgenn
^^^^^

.. automodule:: rotorch.models.cgenn
   :members:
   :imported-members:
   :show-inheritance:

GATr
^^^^

.. automodule:: rotorch.models.gatr
   :members:
   :imported-members:
   :show-inheritance:
