#!/bin/bash

#SBATCH -A XXXXX              # account name
#SBATCH -C cpu                # constraint
#SBATCH --qos=regular         # queue type [regular, debug, etc.]
#SBATCH --time=24:00:00       # wall clock time limit (format: hh:mm:ss)
#SBATCH --nodes=1             # number of nodes
#SBATCH --ntasks-per-node=64  # number of tasks per node

##################################################################################
# Strong scaling: fixed problem size (60x100x2 = 12,000 elements),
# increasing number of MPI ranks. Each rank count is repeated NUM_RUNS times.
#
# For clean timing, this script generates a "timing" variant of the benchmark
# XML with NO VTK output event (I/O would pollute the measurements). The
# Restart block in the base XML is never triggered (no event targets it).
#
# CPU-only nodes: 128 physical cores, 2 hardware threads each (256 logical).
# -c is set to 2 * floor(128 / tasks_per_node).
#
# Partitioning is done along y (-y N) because:
#   - y has 100 uniform elements (enough for up to 64 partitions)
#   - x only has 60 biased elements and the fracture plane sits at x = 0
##################################################################################

GEOS_BIN="PATH_TO_GEOS/bin/geosx"
XML_DIR="PATH_TO_GEOS/inputs/xml_folder"
OUTPUT_DIR_NAME="PATH_TO_OUTPUTS/strong_scaling_kgd"

NUM_RUNS=10

echo "job name: kgdToughnessDominated_benchmark strong scaling"
echo "xml dir: $XML_DIR"

echo "loading modules..."

module load gcc-native/13.2
module load openmpi/5.0.7
module load cmake/3.24.3

echo "activating python env..."

source PATH_TO_PYTHON_ENV/bin/activate

echo "changing dir..."

if [ ! -d "$OUTPUT_DIR_NAME" ]; then
  echo "Creating output directory: $OUTPUT_DIR_NAME"
  mkdir -p $OUTPUT_DIR_NAME
else
  echo "Output directory already exists: $OUTPUT_DIR_NAME"
fi

cd $OUTPUT_DIR_NAME

echo "current dir: $(pwd)"

##################################################################################
# Generate the timing XML: identical to the benchmark, but WITHOUT the VTK
# output PeriodicEvent.
##################################################################################
XML_FILE="${XML_DIR}/kgdToughnessDominated_benchmark_timing.xml"

cat > "$XML_FILE" << EOF
<?xml version="1.0" ?>

<!-- Auto-generated timing variant of kgdToughnessDominated_benchmark.xml:
     identical mesh/events, VTK output event removed for clean timing. -->

<Problem>
  <Included>
    <File name="./kgdToughnessDominated_base.xml"/>
  </Included>

  <Mesh>
    <InternalMesh
      name="mesh1"
      elementTypes="{C3D8}"
      xCoords="{ -100, 0, 100 }"
      yCoords="{ 0, 50 }"
      zCoords="{ 0, 1 }"
      nx="{ 30, 30 }"
      ny="{ 100 }"
      nz="{ 2 }"
      xBias="{ 0.5, -0.5 }"
      cellBlockNames="{cb1}"/>
  </Mesh>

  <Geometry>
    <Box
      name="fracture"
      xMin="{ -0.01, -0.01, -0.01 }"
      xMax="{  0.01,  1.01,  1.01 }"/>

    <Box
      name="source"
      xMin="{ -0.01, -0.01, -0.01 }"
      xMax="{  0.01,  1.01,  1.01 }"/>

    <Box
      name="core"
      xMin="{ -0.01, -0.01, -0.01 }"
      xMax="{  0.01, 50.01,  1.01 }"/>
  </Geometry>

  <Events
    maxTime="100.1">

    <SoloEvent
      name="preFracture"
      target="/Solvers/SurfaceGen"/>

    <PeriodicEvent
      name="solverApplications0"
      beginTime="0.0"
      endTime="10.0"
      forceDt="0.25"
      target="/Solvers/hydrofracture"/>

    <PeriodicEvent
      name="solverApplications1"
      beginTime="10.0"
      endTime="20.0"
      forceDt="0.5"
      target="/Solvers/hydrofracture"/>

    <PeriodicEvent
      name="solverApplications2"
      beginTime="20.0"
      forceDt="1.0"
      target="/Solvers/hydrofracture"/>
  </Events>
</Problem>
EOF

echo "timing xml generated: $XML_FILE"

##################################################################################
# Run loop
##################################################################################
for run in $(seq 1 $NUM_RUNS); do
  echo "==================== Run $run / $NUM_RUNS ===================="

  for i in 64 32 16 8 4 2 1; do
    echo "Running with $i processes (run $run)"

    _output_dir="${OUTPUT_DIR_NAME}/kgd_benchmark_${i}_procs_run${run}"
    _log_file="${_output_dir}/output_run${run}.log"

    mkdir -p $_output_dir
    touch $_log_file

    echo "Output directory: $_output_dir"

    # CPUs per task: 2 * floor(128 / tasks_per_node)
    _cpus_per_task=$((2 * (128 / i)))

    srun -n $i --cpu-bind=cores -c $_cpus_per_task $GEOS_BIN -i $XML_FILE -y $i -o $_output_dir > $_log_file 2>&1
  done
done

echo "$( date ): End loop"
echo "Job completed"

