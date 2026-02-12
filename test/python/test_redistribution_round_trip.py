# -----------------------------------------------------------------------------
# Test CLM restart redistribution round-trip
# This test verifies that redistribution is reversible by:
# 1. Extracting tile ordering from 1x1 and 2x2 restart files
# 2. Redistributing 1x1 -> 2x2 using 2x2 tile ordering
# 3. Redistributing 2x2 -> 1x1 using 1x1 tile ordering
# 4. Comparing original 1x1 file with round-trip result byte-by-byte
# -----------------------------------------------------------------------------

import sys
import os
import glob
from pathlib import Path
from parflow.tools.clm_restart import redistribute_clm_restart, CLMRestartReader

# Test configuration
run_name = "clm_restart_real_data"
nlevsoi = 4
nlevsno = 5

# Directory paths (hard-coded for now)
base_test_output_dir = "/home/ga6/workspace/parflow/build/test/python/test_output"
restart_first_dir = os.path.join(base_test_output_dir, f"{run_name}_restart_first24h")
reference_2x2_dir = os.path.join(base_test_output_dir, f"{run_name}_reference_2x2")
round_trip_dir = os.path.join(base_test_output_dir, f"{run_name}_round_trip")
intermediate_2x2_dir = os.path.join(round_trip_dir, "intermediate_2x2")
final_1x1_dir = os.path.join(round_trip_dir, "final_1x1")
tile_ordering_dir = os.path.join(round_trip_dir, "tile_ordering")

# Create directories
os.makedirs(round_trip_dir, exist_ok=True)
os.makedirs(intermediate_2x2_dir, exist_ok=True)
os.makedirs(final_1x1_dir, exist_ok=True)
os.makedirs(tile_ordering_dir, exist_ok=True)

print("=" * 70)
print("CLM Restart Redistribution Round-Trip Test")
print("=" * 70)

# -----------------------------------------------------------------------------
# Step 1: Extract tile ordering from 1x1 restart file
# -----------------------------------------------------------------------------

print("\nStep 1: Extracting tile ordering from 1x1 restart file...")

# Find the 1x1 restart file
original_1x1_files = glob.glob(os.path.join(restart_first_dir, 'clm.rst.00000.0'))
if not original_1x1_files:
    raise RuntimeError(f"No 1x1 restart file found in {restart_first_dir}")

original_1x1_file = original_1x1_files[0]
print(f"  Found 1x1 restart file: {os.path.basename(original_1x1_file)}")

# Read the 1x1 restart file and extract tile ordering
reader = CLMRestartReader(nlevsoi, nlevsno)
original_1x1_data = reader.read(original_1x1_file)

# Write col_new and row_new for 1x1 topology
col_new_1x1_file = os.path.join(tile_ordering_dir, 'col_new_1x1.txt')
row_new_1x1_file = os.path.join(tile_ordering_dir, 'row_new_1x1.txt')

with open(col_new_1x1_file, 'w') as f:
    f.write(' '.join(map(str, original_1x1_data['col'])) + '\n')

with open(row_new_1x1_file, 'w') as f:
    f.write(' '.join(map(str, original_1x1_data['row'])) + '\n')

print(f"  Extracted 1x1 tile ordering: {len(original_1x1_data['col'])} tiles")
print(f"  Written to: {col_new_1x1_file}")
print(f"  Written to: {row_new_1x1_file}")

# Get domain size from the restart file
# We need to infer nx, ny from the col/row arrays
max_col = max(original_1x1_data['col'])
max_row = max(original_1x1_data['row'])
# Note: col/row are 1-indexed in CLM, so max values give us nx, ny
nx = max_col
ny = max_row

print(f"  Inferred domain size: nx={nx}, ny={ny}")

# -----------------------------------------------------------------------------
# Step 2: Extract tile ordering from 2x2 restart files
# -----------------------------------------------------------------------------

print("\nStep 2: Extracting tile ordering from 2x2 restart files...")

# Find the 2x2 restart files
col_new_2x2_file = os.path.join(tile_ordering_dir, 'col_new_2x2.txt')
row_new_2x2_file = os.path.join(tile_ordering_dir, 'row_new_2x2.txt')

nranks_2x2 = 4  # 2x2 = 4 ranks
with open(col_new_2x2_file, 'w') as col_f, open(row_new_2x2_file, 'w') as row_f:
    for rank in range(nranks_2x2):
        rst_file = os.path.join(reference_2x2_dir, f'clm.rst.00000.{rank}')
        if not os.path.exists(rst_file):
            raise RuntimeError(f"2x2 restart file not found: {rst_file}")
        
        data = reader.read(rst_file)
        col_f.write(' '.join(map(str, data['col'])) + '\n')
        row_f.write(' '.join(map(str, data['row'])) + '\n')
        print(f"  Extracted from rank {rank}: {len(data['col'])} tiles")

print(f"  Written to: {col_new_2x2_file}")
print(f"  Written to: {row_new_2x2_file}")

# -----------------------------------------------------------------------------
# Step 3: Redistribute 1x1 -> 2x2 using 2x2 tile ordering
# -----------------------------------------------------------------------------

print("\nStep 3: Redistributing 1x1 -> 2x2 using 2x2 tile ordering...")

redistribute_clm_restart(
    nx=nx,
    ny=ny,
    old_P=1,
    old_Q=1,
    new_P=2,
    new_Q=2,
    old_restart_dir=restart_first_dir,
    new_restart_dir=intermediate_2x2_dir,
    restart_prefix='clm.rst.',
    tstamp=0,
    nlevsoi=nlevsoi,
    nlevsno=nlevsno,
    col_row_file=(col_new_2x2_file, row_new_2x2_file)
)

print(f"  Redistributed files written to: {intermediate_2x2_dir}")

# Verify intermediate 2x2 files exist
for rank in range(nranks_2x2):
    rst_file = os.path.join(intermediate_2x2_dir, f'clm.rst.00000.{rank}')
    if not os.path.exists(rst_file):
        raise RuntimeError(f"Intermediate 2x2 file not created: {rst_file}")
    file_size = os.path.getsize(rst_file)
    print(f"  Rank {rank}: {os.path.basename(rst_file)} ({file_size:,} bytes)")

# -----------------------------------------------------------------------------
# Step 4: Redistribute 2x2 -> 1x1 using 1x1 tile ordering
# -----------------------------------------------------------------------------

print("\nStep 4: Redistributing 2x2 -> 1x1 using 1x1 tile ordering...")

redistribute_clm_restart(
    nx=nx,
    ny=ny,
    old_P=2,
    old_Q=2,
    new_P=1,
    new_Q=1,
    old_restart_dir=intermediate_2x2_dir,
    new_restart_dir=final_1x1_dir,
    restart_prefix='clm.rst.',
    tstamp=0,
    nlevsoi=nlevsoi,
    nlevsno=nlevsno,
    col_row_file=(col_new_1x1_file, row_new_1x1_file)
)

print(f"  Redistributed files written to: {final_1x1_dir}")

# Verify final 1x1 file exists
final_1x1_file = os.path.join(final_1x1_dir, 'clm.rst.00000.0')
if not os.path.exists(final_1x1_file):
    raise RuntimeError(f"Final 1x1 file not created: {final_1x1_file}")
file_size = os.path.getsize(final_1x1_file)
print(f"  Final file: {os.path.basename(final_1x1_file)} ({file_size:,} bytes)")

# -----------------------------------------------------------------------------
# Step 5: Compare original and round-trip files byte-by-byte
# -----------------------------------------------------------------------------

print("\nStep 5: Comparing original and round-trip files byte-by-byte...")

original_size = os.path.getsize(original_1x1_file)
final_size = os.path.getsize(final_1x1_file)

print(f"  Original file size: {original_size:,} bytes")
print(f"  Final file size:    {final_size:,} bytes")

if original_size != final_size:
    print(f"  FAILED: File sizes differ by {abs(original_size - final_size):,} bytes")
    sys.exit(1)

# Byte-by-byte comparison
print("  Performing byte-by-byte comparison...")
with open(original_1x1_file, 'rb') as f_orig, open(final_1x1_file, 'rb') as f_final:
    byte_count = 0
    diff_count = 0
    first_diff_pos = None
    
    while True:
        byte_orig = f_orig.read(1)
        byte_final = f_final.read(1)
        
        if not byte_orig and not byte_final:
            break
        
        if not byte_orig or not byte_final:
            print(f"  FAILED: Files have different lengths")
            sys.exit(1)
        
        byte_count += 1
        if byte_orig != byte_final:
            diff_count += 1
            if first_diff_pos is None:
                first_diff_pos = byte_count
                print(f"  First difference at byte position: {first_diff_pos}")
                print(f"    Original: 0x{byte_orig.hex()}")
                print(f"    Final:    0x{byte_final.hex()}")
    
    if diff_count == 0:
        print(f"  SUCCESS: Files are identical ({byte_count:,} bytes compared)")
    else:
        print(f"  FAILED: {diff_count:,} bytes differ out of {byte_count:,} total")
        if diff_count <= 10:
            # Show all differences if there are few
            print("  Showing all differences...")
            f_orig.seek(0)
            f_final.seek(0)
            pos = 0
            while True:
                byte_orig = f_orig.read(1)
                byte_final = f_final.read(1)
                if not byte_orig:
                    break
                pos += 1
                if byte_orig != byte_final:
                    print(f"    Byte {pos}: 0x{byte_orig.hex()} != 0x{byte_final.hex()}")
        sys.exit(1)

# -----------------------------------------------------------------------------
# Additional verification: Read and compare data structures
# -----------------------------------------------------------------------------

print("\nStep 6: Verifying data structure integrity...")

original_data = reader.read(original_1x1_file)
final_data = reader.read(final_1x1_file)

# Compare key fields
comparisons = [
    ('nch', original_data['nch'], final_data['nch']),
    ('nc', original_data['nc'], final_data['nc']),
    ('nr', original_data['nr'], final_data['nr']),
    ('col', original_data['col'], final_data['col']),
    ('row', original_data['row'], final_data['row']),
]

all_match = True
for name, orig_val, final_val in comparisons:
    if isinstance(orig_val, (list, tuple)) or hasattr(orig_val, '__array__'):
        import numpy as np
        if np.array_equal(orig_val, final_val):
            print(f"  {name}: MATCH")
        else:
            print(f"  {name}: MISMATCH")
            all_match = False
    else:
        if orig_val == final_val:
            print(f"  {name}: MATCH ({orig_val})")
        else:
            print(f"  {name}: MISMATCH ({orig_val} != {final_val})")
            all_match = False

if not all_match:
    print("  WARNING: Some data fields differ (but files are byte-identical)")
else:
    print("  All data fields match")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Round-Trip Test Summary")
print("=" * 70)
print("✓ Extracted tile ordering from 1x1 restart file")
print("✓ Extracted tile ordering from 2x2 restart files")
print("✓ Redistributed 1x1 -> 2x2 using 2x2 tile ordering")
print("✓ Redistributed 2x2 -> 1x1 using 1x1 tile ordering")
print("✓ Byte-by-byte comparison: PASSED")
if all_match:
    print("✓ Data structure verification: PASSED")
print("=" * 70)
print("Round-trip test completed successfully!")
print("=" * 70)
