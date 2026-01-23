.. _api_ref:

.. currentmodule:: neuromodes

API Reference
=============

.. contents:: **List of modules**
   :local:

.. _ref_eigen:

:mod:`neuromodes.eigen` - Compute eigenmodes on cortical surface meshes
---------------------------------------------------------------

.. automodule:: neuromodes.eigen
   :no-members:
   :no-inherited-members:

.. currentmodule:: neuromodes.eigen

.. autosummary::
   :template: class.rst
   :toctree: generated/

   EigenSolver

.. autoclass:: EigenSolver
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :template: function.rst
   :toctree: generated/

   neuromodes.eigen.EigenSolver.compute_lbo
   neuromodes.eigen.EigenSolver.solve
   neuromodes.eigen.EigenSolver.decompose
   neuromodes.eigen.EigenSolver.reconstruct
   neuromodes.eigen.EigenSolver.reconstruct_timeseries
   neuromodes.eigen.EigenSolver.simulate_waves
   neuromodes.eigen.EigenSolver.model_connectome

.. autosummary::
   :template: function.rst
   :toctree: generated/

   neuromodes.eigen.scale_hetero
   neuromodes.eigen.standardize_modes
   neuromodes.eigen.is_orthonormal_basis

.. _ref_basis:

:mod:`neuromodes.basis` - Decompose and reconstruct cortical maps
---------------------------------------------------------------

.. automodule:: neuromodes.basis
   :no-members:
   :no-inherited-members:

.. currentmodule:: neuromodes.basis

.. autosummary::
   :template: function.rst
   :toctree: generated/

   neuromodes.basis.decompose
   neuromodes.basis.reconstruct
   neuromodes.basis.reconstruct_timeseries
   neuromodes.basis.calc_norm_power
   neuromodes.basis.calc_vec_fc

.. _ref_waves:

:mod:`neuromodes.waves` - Simulate activity via wave propagation
---------------------------------------------------------------

.. automodule:: neuromodes.waves
   :no-members:
   :no-inherited-members:

.. currentmodule:: neuromodes.waves

.. autosummary::
   :template: function.rst
   :toctree: generated/

   neuromodes.waves.simulate_waves
   neuromodes.waves.calc_wave_speed
   neuromodes.waves.get_balloon_params

.. _ref_connectome:

:mod:`neuromodes.connectome` - Generative modelling of the structural connectome
---------------------------------------------------------------

.. automodule:: neuromodes.connectome
   :no-members:
   :no-inherited-members:

.. currentmodule:: neuromodes.connectome

.. autosummary::
   :template: function.rst
   :toctree: generated/

   neuromodes.connectome.model_connectome

.. _ref_io:

:mod:`neuromodes.io` - IO functions for cortical surface meshes and maps
---------------------------------------------------------------

.. automodule:: neuromodes.io
   :no-members:
   :no-inherited-members:

.. currentmodule:: neuromodes.io

.. autosummary::
   :template: function.rst
   :toctree: generated/

   neuromodes.io.is_vol
   neuromodes.io.is_surf
   neuromodes.io.read_vol
   neuromodes.io.read_surf
   neuromodes.io.mask_surf
   neuromodes.io.check_vol
   neuromodes.io.check_surf
   neuromodes.io.fetch_vol
   neuromodes.io.fetch_surf
   neuromodes.io.fetch_map