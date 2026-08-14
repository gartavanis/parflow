PFTools API Reference
=====================

Each entry below includes a short description from the object's docstring.
Click a name to open the full documentation.

.. toctree::
   :maxdepth: 1
   :hidden:

   parflow
   run
   hydrology
   fs

.. rubric:: Parflow Module

.. currentmodule:: parflow

.. autosummary::
   :nosignatures:

   ParflowBinaryReader
   Run
   read_pfb
   write_pfb
   read_pfb_sequence
   pf_test_file
   pf_test_file_with_abs

.. rubric:: Run Class

.. currentmodule:: parflow.tools.core

.. autosummary::
   :nosignatures:

   Run.from_definition
   Run.get_name
   Run.set_name
   Run.write
   Run.write_subsurface_table
   Run.clone
   Run.run
   Run.check_nans
   Run.dist
   Run.undist
   Run.details
   Run.doc
   Run.get_children_of_type
   Run.get_context_settings
   Run.keys
   Run.pfset
   Run.select
   Run.to_dict
   Run.to_pf_name
   Run.validate
   Run.value
   Run.data_accessor
   Run.full_name

.. rubric:: Hydrology Module

.. currentmodule:: parflow.tools.hydrology

.. autosummary::
   :nosignatures:

   calculate_evapotranspiration
   calculate_overland_flow
   calculate_overland_flow_grid
   calculate_overland_fluxes
   calculate_subsurface_storage
   calculate_surface_storage
   calculate_water_table_depth
   compute_hydraulic_head
   compute_water_table_depth

.. rubric:: FS Module

.. currentmodule:: parflow.tools.fs

.. autosummary::
   :nosignatures:

   cp
   get_absolute_path
   mkdir
   rm
   get_text_file_content
   exists
   chdir
