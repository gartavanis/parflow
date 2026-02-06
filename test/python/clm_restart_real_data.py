# -----------------------------------------------------------------------------
# Test CLM restart with real data from subsettools
# This test uses real domain data created using the subsettools package
# -----------------------------------------------------------------------------

import sys
import os
import glob
from pathlib import Path
from parflow import Run
from parflow.tools.fs import mkdir, cp, get_absolute_path
from parflow.tools.clm_restart import redistribute_clm_restart
from parflow.tools.compare import pf_test_file

# -----------------------------------------------------------------------------
# Create run directory and copy input files
# -----------------------------------------------------------------------------

# Dummy run to get paths straight
run_name = "clm_restart_real_data"
run = Run(run_name, __file__)
run_dir = get_absolute_path(f"test_output/{run_name}")
mkdir(run_dir)

# Copy all input files from test/input/clm_restart_inputs
input_dir = get_absolute_path("$PF_SRC/test/input/clm_restart_inputs")

# Copy all files from input directory to run directory
input_path = Path(input_dir)
for file_path in input_path.iterdir():
    if file_path.is_file():
        cp(str(file_path), run_dir)

# The yaml file is now in the run directory
runscript_path = os.path.join(run_dir, "clm_restart_run.yaml")

# -----------------------------------------------------------------------------
# Load run from YAML definition
# -----------------------------------------------------------------------------

run = Run.from_definition(runscript_path)
run.set_name("original_run")

# -----------------------------------------------------------------------------
# Set the required keys
# -----------------------------------------------------------------------------

run.TimingInfo.StopTime = 48
run.Solver.CLM.MetFileName = 'CW3E'
run.Solver.CLM.MetFilePath = '.'
run.Solver.Linear.Preconditioner = "PFMGOctree"

# -----------------------------------------------------------------------------
# Distribute PFB files that are referenced in the YAML
# -----------------------------------------------------------------------------

# List of PFB files that need to be distributed (excluding .dist files and forcing files)
pfb_files_to_distribute = [
    'slope_x.pfb',
    'slope_y.pfb',
    'mannings.pfb',
    'pf_flowbarrier.pfb',
    'pf_indicator.pfb',
    'ss_pressure_head.pfb',
    'mask.pfb',
]

# Distribute each PFB file
for pfb_file in pfb_files_to_distribute:
    pfb_path = os.path.join(run_dir, pfb_file)
    if os.path.exists(pfb_path):
        run.dist(pfb_path)

# Distribute CW3E forcing files (these are time-series files)
cw3e_files = [
    'CW3E.APCP.000001_to_000024.pfb',
    'CW3E.APCP.000025_to_000048.pfb',
    'CW3E.DLWR.000001_to_000024.pfb',
    'CW3E.DLWR.000025_to_000048.pfb',
    'CW3E.DSWR.000001_to_000024.pfb',
    'CW3E.DSWR.000025_to_000048.pfb',
    'CW3E.Press.000001_to_000024.pfb',
    'CW3E.Press.000025_to_000048.pfb',
    'CW3E.SPFH.000001_to_000024.pfb',
    'CW3E.SPFH.000025_to_000048.pfb',
    'CW3E.Temp.000001_to_000024.pfb',
    'CW3E.Temp.000025_to_000048.pfb',
    'CW3E.UGRD.000001_to_000024.pfb',
    'CW3E.UGRD.000025_to_000048.pfb',
    'CW3E.VGRD.000001_to_000024.pfb',
    'CW3E.VGRD.000025_to_000048.pfb',
]

for cw3e_file in cw3e_files:
    cw3e_path = os.path.join(run_dir, cw3e_file)
    if os.path.exists(cw3e_path):
        run.dist(cw3e_path)

# -----------------------------------------------------------------------------
# Run ParFlow
# -----------------------------------------------------------------------------

run.run(working_directory=run_dir)

print(f"{run_name} : 48h simulation complete")

# -----------------------------------------------------------------------------
# Restart run: Run 24h, redistribute to 2x2, then continue to 48h
# -----------------------------------------------------------------------------

# Get domain size and CLM layer info from original run
nx = run.ComputationalGrid.NX
ny = run.ComputationalGrid.NY
nlevsoi = 4
nlevsno = 0

# Create first part of restart run (24 hours with 1x1 topology)
restart_first_dir = get_absolute_path(f"test_output/{run_name}_restart_first24h")
mkdir(restart_first_dir)

# Copy all input files to restart first directory
for file_path in input_path.iterdir():
    if file_path.is_file():
        cp(str(file_path), restart_first_dir)

# Create first part of restart run from YAML
restart_first_script_path = os.path.join(restart_first_dir, "clm_restart_run.yaml")
restart_first_run = Run.from_definition(restart_first_script_path)
restart_first_run.set_name("restart_run")

# Set keys for first 24 hours
restart_first_run.TimingInfo.StopTime = 24
restart_first_run.Solver.CLM.MetFileName = 'CW3E'
restart_first_run.Solver.CLM.MetFilePath = '.'
restart_first_run.Solver.Linear.Preconditioner = "PFMGOctree"
restart_first_run.Process.Topology.P = 1
restart_first_run.Process.Topology.Q = 1
restart_first_run.Process.Topology.R = 1

# Distribute PFB files for first part
for pfb_file in pfb_files_to_distribute:
    pfb_path = os.path.join(restart_first_dir, pfb_file)
    if os.path.exists(pfb_path):
        restart_first_run.dist(pfb_path)

# Distribute CW3E forcing files for first 24 hours
cw3e_files_first24 = [
    'CW3E.APCP.000001_to_000024.pfb',
    'CW3E.DLWR.000001_to_000024.pfb',
    'CW3E.DSWR.000001_to_000024.pfb',
    'CW3E.Press.000001_to_000024.pfb',
    'CW3E.SPFH.000001_to_000024.pfb',
    'CW3E.Temp.000001_to_000024.pfb',
    'CW3E.UGRD.000001_to_000024.pfb',
    'CW3E.VGRD.000001_to_000024.pfb',
]

for cw3e_file in cw3e_files_first24:
    cw3e_path = os.path.join(restart_first_dir, cw3e_file)
    if os.path.exists(cw3e_path):
        restart_first_run.dist(cw3e_path)

# Run first 24 hours
print("Running restart run: first 24 hours...")
restart_first_run.run(working_directory=restart_first_dir)
print("First 24 hours complete")

# -----------------------------------------------------------------------------
# Redistribute CLM restart files from 1x1 to 2x2 topology
# -----------------------------------------------------------------------------

restart_timestamp = 24
old_restart_dir = restart_first_dir
new_restart_dir = get_absolute_path(f"test_output/{run_name}_restart_2x2")
mkdir(new_restart_dir)

print("Redistributing CLM restart files from 1x1 to 2x2 topology...")
redistribute_clm_restart(
    nx=nx,
    ny=ny,
    old_P=1,
    old_Q=1,
    new_P=2,
    new_Q=2,
    old_restart_dir=old_restart_dir,
    new_restart_dir=new_restart_dir,
    restart_prefix='clm.rst.',
    tstamp=0,
    nlevsoi=nlevsoi,
    nlevsno=nlevsno,
    col_row_file=None  # Use sequential ordering
)
print("CLM restart files redistributed")

# -----------------------------------------------------------------------------
# Create second part of restart run (24-48 hours with 2x2 topology)
# -----------------------------------------------------------------------------

restart_second_dir = get_absolute_path(f"test_output/{run_name}_restart_second24h")
mkdir(restart_second_dir)

# Copy all input files to restart second directory
for file_path in input_path.iterdir():
    if file_path.is_file():
        cp(str(file_path), restart_second_dir)

# Rename drv_clmin_restart.dat to drv_clmin.dat for restart
drv_clmin_restart_path = os.path.join(restart_second_dir, "drv_clmin_restart.dat")
drv_clmin_path = os.path.join(restart_second_dir, "drv_clmin.dat")
if os.path.exists(drv_clmin_restart_path):
    os.rename(drv_clmin_restart_path, drv_clmin_path)
    print(f"Renamed drv_clmin_restart.dat to drv_clmin.dat in restart directory")

# Copy redistributed restart files
for rst_file in glob.glob(os.path.join(new_restart_dir, 'clm.rst.*')):
    cp(rst_file, restart_second_dir)

# Rename restart files from timestamp 0 to restart_timestamp (24)
# ParFlow expects restart files with the timestep number when restarting
for rst_file in glob.glob(os.path.join(restart_second_dir, 'clm.rst.00000.*')):
    new_name = rst_file.replace('clm.rst.00000.', f'clm.rst.{restart_timestamp:05d}.')
    os.rename(rst_file, new_name)
    print(f"Renamed {os.path.basename(rst_file)} to {os.path.basename(new_name)}")

# Create second part of restart run from YAML
restart_second_script_path = os.path.join(restart_second_dir, "clm_restart_run.yaml")
restart_second_run = Run.from_definition(restart_second_script_path)
restart_second_run.set_name("restart_run")

# Set topology to 2x2
restart_second_run.Process.Topology.P = 2
restart_second_run.Process.Topology.Q = 2
restart_second_run.Process.Topology.R = 1

# Set timing for restart (start from step 24, run to step 48)
restart_second_run.TimingInfo.StartCount = restart_timestamp
restart_second_run.TimingInfo.StartTime = float(restart_timestamp)
restart_second_run.TimingInfo.StopTime = 48

# Set CLM restart parameters
restart_second_run.Solver.CLM.IstepStart = restart_timestamp + 1
restart_second_run.Solver.CLM.MetFileName = 'CW3E'
restart_second_run.Solver.CLM.MetFilePath = '.'
restart_second_run.Solver.Linear.Preconditioner = "PFMGOctree"

# Distribute PFB files for restart second part (with 2x2 topology)
for pfb_file in pfb_files_to_distribute:
    pfb_path = os.path.join(restart_second_dir, pfb_file)
    if os.path.exists(pfb_path):
        restart_second_run.dist(pfb_path)

# Distribute CW3E forcing files for hours 25-48
cw3e_files_second24 = [
    'CW3E.APCP.000025_to_000048.pfb',
    'CW3E.DLWR.000025_to_000048.pfb',
    'CW3E.DSWR.000025_to_000048.pfb',
    'CW3E.Press.000025_to_000048.pfb',
    'CW3E.SPFH.000025_to_000048.pfb',
    'CW3E.Temp.000025_to_000048.pfb',
    'CW3E.UGRD.000025_to_000048.pfb',
    'CW3E.VGRD.000025_to_000048.pfb',
]

for cw3e_file in cw3e_files_second24:
    cw3e_path = os.path.join(restart_second_dir, cw3e_file)
    if os.path.exists(cw3e_path):
        restart_second_run.dist(cw3e_path)

# Handle pressure restart file from first part
pressure_restart_file = os.path.join(restart_first_dir, f"{restart_first_run.get_name()}.out.press.{restart_timestamp:05d}.pfb")
if os.path.exists(pressure_restart_file):
    # Copy the pressure file and redistribute it for 2x2 topology
    pressure_ic_file = os.path.join(restart_second_dir, "press_restart.pfb")
    cp(pressure_restart_file, pressure_ic_file)
    # Redistribute for 2x2 topology
    restart_second_run.dist(pressure_ic_file)
    # Update the initial condition to use this file
    restart_second_run.ICPressure.Type = 'PFBFile'
    restart_second_run.Geom.domain.ICPressure.FileName = 'press_restart.pfb'

# Run second 24 hours
print("Running restart run: second 24 hours (24-48h) with 2x2 topology...")
restart_second_run.run(working_directory=restart_second_dir)
print("Second 24 hours complete")

# -----------------------------------------------------------------------------
# Compare outputs from original_run and restart_run
# -----------------------------------------------------------------------------

print("Comparing outputs from original_run and restart_run...")
passed = True

# Fields to compare
fields_to_compare = ['press', 'satur']

# Compare timesteps 0-24: original_run vs restart_first_run
print("Comparing timesteps 0-24: original_run vs restart_first_run...")
for timestep in range(0, 25):  # 0 to 24 inclusive
    timestep_str = f"{timestep:05d}"
    for field in fields_to_compare:
        original_file = os.path.join(run_dir, f"{run.get_name()}.out.{field}.{timestep_str}.pfb")
        restart_file = os.path.join(restart_first_dir, f"{restart_first_run.get_name()}.out.{field}.{timestep_str}.pfb")
        
        if os.path.exists(original_file) and os.path.exists(restart_file):
            if not pf_test_file(
                restart_file,
                original_file,
                f"Max difference in {field} at timestep {timestep}"
            ):
                passed = False
                print(f"  FAILED: {field} comparison at timestep {timestep}")
            # Only print success for every 5th timestep to reduce output
            elif timestep % 5 == 0:
                print(f"  PASSED: {field} comparison at timestep {timestep}")
        elif timestep == 0 and field == 'press':
            # Initial condition might not exist as output
            pass
        else:
            print(f"  WARNING: Could not find {field} files for timestep {timestep}")

# Compare timesteps 24-48: original_run vs restart_second_run
print("Comparing timesteps 24-48: original_run vs restart_second_run...")
for timestep in range(24, 49):  # 24 to 48 inclusive
    timestep_str = f"{timestep:05d}"
    for field in fields_to_compare:
        original_file = os.path.join(run_dir, f"{run.get_name()}.out.{field}.{timestep_str}.pfb")
        restart_file = os.path.join(restart_second_dir, f"{restart_second_run.get_name()}.out.{field}.{timestep_str}.pfb")
        
        if os.path.exists(original_file) and os.path.exists(restart_file):
            if not pf_test_file(
                restart_file,
                original_file,
                f"Max difference in {field} at timestep {timestep}"
            ):
                passed = False
                print(f"  FAILED: {field} comparison at timestep {timestep}")
            # Only print success for every 5th timestep to reduce output
            elif timestep % 5 == 0:
                print(f"  PASSED: {field} comparison at timestep {timestep}")
        else:
            print(f"  WARNING: Could not find {field} files for timestep {timestep}")

if passed:
    print(f"{run_name} : ALL TESTS PASSED")
else:
    print(f"{run_name} : SOME TESTS FAILED")
    sys.exit(1)