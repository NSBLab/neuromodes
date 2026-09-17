.. _api_ref:

.. currentmodule:: neuromodes

API Reference
=============

.. contents:: **List of modules**
   :local:

.. _ref_eigen:

:mod:`neuromodes.eigen` - Compute geometric eigenmodes of cortical surface meshes
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
   recon_error
   sim_nft_waves
   balloon_model
   compute_gem
   eigenstrap
   unmask_data

.. autosummary::
   :template: function.rst
   :toctree: generated/

   align_basis
   is_orthonormal_basis
   get_eigengroup_inds

.. _ref_basis:

:mod:`neuromodes.basis` - Decompose and reconstruct brain maps
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
   recon_error

.. _ref_waves:

:mod:`neuromodes.waves` - Simulate neural activity and functional MRI signals
---------------------------------------------------------------

.. automodule:: neuromodes.waves
   :no-members:
   :no-inherited-members:

.. currentmodule:: neuromodes.waves

.. autosummary::
   :template: function.rst
   :toctree: generated/

   sim_nft_waves
   calc_nft_wave_speed
   calc_nft_mode_freqs
   calc_nft_fc
   balloon_model

.. _ref_network:

:mod:`neuromodes.network` - Generative modelling of structural connectivity
---------------------------------------------------------------

.. automodule:: neuromodes.network
   :no-members:
   :no-inherited-members:

.. currentmodule:: neuromodes.network

.. autosummary::
   :template: function.rst
   :toctree: generated/

   compute_gem

.. _ref_stats:

:mod:`neuromodes.stats` - Statistical functions for brain map analysis
---------------------------------------------------------------

.. automodule:: neuromodes.stats
   :no-members:
   :no-inherited-members:

.. currentmodule:: neuromodes.stats

.. autosummary::
   :template: function.rst
   :toctree: generated/

   gramw
   dotw
   ssqw
   meanw
   demeanw
   varw
   momentw
   stdw
   zscorew
   covw
   vecnormw
   cdistw
   pdistw
   solvew
   lstsqw
   parcellate
   sigmoid_rescale

.. _ref_mesh:

:mod:`neuromodes.mesh` - Mask and validate meshes of brain structures
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

:mod:`neuromodes.nulls` - Generate null brain maps
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
   fetch_example_surf
   fetch_example_map