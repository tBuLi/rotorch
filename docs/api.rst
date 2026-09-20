API reference
=============

Both :mod:`rotorch.nn` and :mod:`rotorch.models` are organised one subpackage per architecture.
Only cgenn is implemented so far, so everything below lives under ``cgenn``; GATr and
others will appear alongside it rather than in place of it.

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


Testing
-------

.. automodule:: rotorch.testing
   :members:
Models
------

.. automodule:: rotorch.models

cgenn
^^^^^

.. automodule:: rotorch.models.cgenn
   :members:
   :imported-members:
   :show-inheritance:
