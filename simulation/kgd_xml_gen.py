"""Generate KGD spacing-sweep decks from the GEOS "kgdToughnessDominated" example.

Derived from the example distributed with GEOS (inputFiles/hydraulicFracturing):
https://github.com/GEOS-DEV/GEOS/tree/develop/inputFiles/hydraulicFracturing
Generated decks carry the same attribution header as the packaged decks in
inputs/kgd/ (see simulation/upstream_geos_examples/PROVENANCE.md).
"""
# get user input
import sys
if len(sys.argv) != 2:
    print("Usage: python kgdToughnessDominated_benchmark_spacing_gen.py <int_spacing>")
    sys.exit(1)

spacing = int(sys.argv[1])
half_spacing = spacing / 2.0

original_xml = """<?xml version="1.0" ?>

<Problem>
  <Included>
    <File name="./kgdToughnessDominated_base_2.xml"/>
  </Included>

  <Mesh>
    <!-- Sphinx_Mesh_InternalMesh -->
    <InternalMesh 
      name="mesh1"
      elementTypes="{C3D8}"
      xCoords="{ -100, 100 }"
      yCoords="{ 0, 50 }"
      zCoords="{ 0, 1 }"
      nx="{100 }"
      ny="{ 100 }"
      nz="{ 2 }"
      cellBlockNames="{cb1}"/>
    <!-- Sphinx_Mesh_InternalMesh_End -->
  </Mesh>

  <Geometry>

    <!-- Sphinx_Geometry_InitFracture -->
    <Box
      name="fracture1"
      xMin="{ val_1_neg, -0.01, -0.01 }"
      xMax="{ val_1_pos,  1.01,  1.01 }"/>
    <!-- Sphinx_Geometry_InitFracture_End -->

    <!-- Sphinx_Geometry_InjSource -->
    <Box
      name="source1"
      xMin="{ val_1_neg, -0.01, -0.01 }"
      xMax="{ val_1_pos,  1.01,  1.01 }"/>
    <!-- Sphinx_Geometry_InjSource_End -->

    <!-- Sphinx_Geometry_FracturePlane -->
    <Box
      name="core1"
      xMin="{ val_1_neg, -0.01, -0.01 }"
      xMax="{ val_1_pos, 50.01,  1.01 }"/>
    <!-- Sphinx_Geometry_FracturePlane_End -->

        <!-- Sphinx_Geometry_InitFracture -->
    <Box
      name="fracture2"
      xMin="{ val_2_neg, -0.01, -0.01 }"
      xMax="{ val_2_pos,  1.01,  1.01 }"/>
    <!-- Sphinx_Geometry_InitFracture_End -->

    <!-- Sphinx_Geometry_InjSource -->
    <Box
      name="source2"
      xMin="{ val_2_neg, -0.01, -0.01 }"
      xMax="{ val_2_pos,  1.01,  1.01 }"/>
    <!-- Sphinx_Geometry_InjSource_End -->

    <!-- Sphinx_Geometry_FracturePlane -->
    <Box
      name="core2"
      xMin="{ val_2_neg, -0.01, -0.01 }"
      xMax="{ val_2_pos, 50.01,  1.01 }"/>
    <!-- Sphinx_Geometry_FracturePlane_End -->
  </Geometry>

</Problem>
"""

val_1_neg = -half_spacing - 0.01
val_1_pos = -half_spacing + 0.01

val_2_neg = half_spacing - 0.01
val_2_pos = half_spacing + 0.01

temp = original_xml.replace("val_1_neg", str(val_1_neg))
temp = temp.replace("val_1_pos", str(val_1_pos))
temp = temp.replace("val_2_neg", str(val_2_neg))
temp = temp.replace("val_2_pos", str(val_2_pos))

file_name = f"kgdToughnessDominated_benchmark_spacing_{spacing}.xml"

with open(file_name, "w") as f:
    f.write(temp)
print(f"Generated {file_name} with spacing {spacing}")