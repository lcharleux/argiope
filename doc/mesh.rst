Mesh
====

Mesh processing tools.

Mesh class
~~~~~~~~~~

.. autoclass:: argiope.mesh.Mesh
  
Set
____________________

.. automethod:: argiope.mesh.Mesh.set_nodes
.. automethod:: argiope.mesh.Mesh.set_elements
.. automethod:: argiope.mesh.Mesh.set_fields

Verify
_______________

.. automethod:: argiope.mesh.Mesh.check_elements
.. automethod:: argiope.mesh.Mesh.nvert
.. automethod:: argiope.mesh.Mesh.centroids_and_volumes
.. automethod:: argiope.mesh.Mesh.angles
.. automethod:: argiope.mesh.Mesh.edges
.. automethod:: argiope.mesh.Mesh.stats
.. automethod:: argiope.mesh.Mesh.fields_metadata


Modify
______________
.. automethod:: argiope.mesh.Mesh.element_set_to_node_set
.. automethod:: argiope.mesh.Mesh.node_set_to_surface
.. automethod:: argiope.mesh.Mesh.surface_to_element_sets

Plot with Matplotlib
_____________________
.. automethod:: argiope.mesh.Mesh.to_polycollection
.. automethod:: argiope.mesh.Mesh.to_triangulation



Export
________
.. automethod:: argiope.mesh.Mesh.write_inp


Fields
========

Meta classes
~~~~~~~~~~~~~~~~
.. autoclass:: argiope.mesh.MetaField
   :members:
   :inherited-members:

.. autoclass:: argiope.mesh.Field
   :members:
   :inherited-members:


Field classes
~~~~~~~~~~~~~~~~~~~

.. autoclass:: argiope.mesh.ScalarField
   :members:
   :inherited-members:

.. autoclass:: argiope.mesh.Vector2Field
   :members:
   :inherited-members:

.. autoclass:: argiope.mesh.Vector3Field
   :members:
   :inherited-members:

.. autoclass:: argiope.mesh.Tensor4Field
   :members:
   :inherited-members:
   
.. autoclass:: argiope.mesh.Tensor6Field
   :members:
   :inherited-members:   
   
Parsers
===========

.. autofunction:: argiope.mesh.read_msh


Mesh generation
===================   

.. autofunction:: argiope.mesh.structured_mesh

