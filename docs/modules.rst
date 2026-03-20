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

   compute_lbo
   solve
   decompose
   reconstruct
   reconstruct_timeseries
   simulate_waves
   bold_transform
   model_connectome
   eigenstrap

.. autosummary::
   :template: function.rst
   :toctree: generated/

   scale_hetero
   standardize_emodes
   is_orthonormal_basis

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

   decompose
   reconstruct
   reconstruct_timeseries
   calc_norm_power
   calc_vec_fc

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

   simulate_waves
   bold_transform
   calc_wave_speed
   get_balloon_params

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

   model_connectome

.. _ref_mesh:

:mod:`neuromodes.mesh` - Create, mask, and validate meshes of brain structures
---------------------------------------------------------------

.. automodule:: neuromodes.mesh
   :no-members:
   :no-inherited-members:

.. currentmodule:: neuromodes.mesh

.. autosummary::
   :template: function.rst
   :toctree: generated/

   mask_mesh
   unmask_data
   check_surf

.. _ref_nulls:

:mod:`neuromodes.nulls` - Generate null brain maps preserving spatial autocorrelation
---------------------------------------------------------------

.. automodule:: neuromodes.nulls
   :no-members:
   :no-inherited-members:

.. currentmodule:: neuromodes.nulls

.. autosummary::
   :template: function.rst
   :toctree: generated/

   eigenstrap

.. _ref_io:

:mod:`neuromodes.io` - IO functions for loading cortical meshes and maps
---------------------------------------------------------------

.. automodule:: neuromodes.io
   :no-members:
   :no-inherited-members:

.. currentmodule:: neuromodes.io

.. autosummary::
   :template: function.rst
   :toctree: generated/

   read_surf
   fetch_surf
   fetch_map