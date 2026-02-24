#!/usr/bin/env python3
"""
Examine CLM restart variables and ET at grid position (i, j) = (42, 82).

Note: ET (evapotranspiration) is NOT stored in CLM restart (.rst) files.
Restart files hold state variables (t_grnd, h2osoi_liq, etc.).
ET is written to CLM output (e.g. qflx_evap_tot) when WriteCLMBinary is true.
This script:
  - Prints all restart variables at (i,j)=(42,82) for dummy and 24h rst files.
  - If CLM binary output exists in the run dir, prints ET (qflx_evap_tot) at (42,82).
"""
import os
import sys
import numpy as np

# Default paths under build/test/python/test_output
BASE = "/home/ga6/workspace/parflow/build/test/python/test_output"
RUN_NAME = "clm_restart_real_data"
# Dummy: initial/main run dir (may have clm.rst at start or we use 24h for both if only one exists)
DUMMY_RST_DIR = os.path.join(BASE, RUN_NAME)
# 24h run: first 24h restart
RST_24H_DIR = os.path.join(BASE, f"{RUN_NAME}_restart_first24h")

# Grid position (1-based column i, row j as in ParFlow/CLM)
I, J = 42, 82

# CLM layer counts (match test)
NLEVSOI = 4
NLEVSNO = 5


def find_tile_at(data, col_val, row_val):
    """Return tile index (0-based) where col==col_val and row==row_val (1-based)."""
    col = data["col"]
    row = data["row"]
    for k in range(len(col)):
        if int(col[k]) == col_val and int(row[k]) == row_val:
            return k
    return None


def read_restart_quiet(filepath, nlevsoi=NLEVSOI, nlevsno=NLEVSNO):
    """Read CLM restart file without printing (patch reader to suppress stdout)."""
    from parflow.tools.clm_restart import CLMRestartReader
    import io
    reader = CLMRestartReader(nlevsoi=nlevsoi, nlevsno=nlevsno)
    # Temporarily suppress the reader's print statements
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        data = reader.read(filepath)
    finally:
        sys.stdout = old_stdout
    return data


def print_restart_at(data, label, col_val, row_val):
    """Print all restart variables for the tile at (col_val, row_val)."""
    k = find_tile_at(data, col_val, row_val)
    if k is None:
        print(f"  No tile with col={col_val}, row={row_val} in {label}")
        return
    print(f"  Tile index (0-based): {k}  (col={col_val}, row={row_val})")
    print(f"  Time: {data['yr']}-{data['mo']:02d}-{data['da']:02d} {data['hr']:02d}:{data['mn']:02d}:{data['ss']:02d}  istep={data['istep']}")
    print("  Scalar fields:")
    for key in ["fgrd", "vegt", "t_grnd", "t_veg", "h2osno", "snowage", "snowdp",
                "h2ocan", "frac_sno", "elai", "esai", "snl", "xerr", "zerr"]:
        if key in data:
            val = data[key][k]
            if isinstance(val, (np.floating, float)):
                print(f"    {key}: {val}")
            else:
                print(f"    {key}: {val}")
    print("  Layer fields (first/last few layers): dz, z, t_soisno, h2osoi_liq, h2osoi_ice")
    nlay = data["h2osoi_liq"].shape[1]
    for name in ["dz", "z", "t_soisno", "h2osoi_liq", "h2osoi_ice"]:
        arr = data[name][k, :]
        print(f"    {name}: min={arr.min()}, max={arr.max()}, [0:3]={arr[:3]}, ... [{-3}:]={arr[-3:]}")


def read_et_from_binary(run_dir, timestep, nc, nr, i, j):
    """
    Read qflx_evap_tot at (i,j) from CLM binary output.
    Binary is written in order j=1..nr, i=1..nc (one value per grid cell).
    """
    # File: qflx_evap_tot.<timestep>.bin.0 (for rank 0, 1x1)
    path = os.path.join(run_dir, f"qflx_evap_tot.{timestep:05d}.bin.0")
    if not os.path.isfile(path):
        return None
    arr = np.fromfile(path, dtype=np.float64)
    if arr.size != nc * nr:
        return None
    # Fortran order: (i,j) with j outer, i inner -> index (j-1)*nc + (i-1)
    idx = (j - 1) * nc + (i - 1)
    return float(arr[idx])


def main():
    print("=" * 60)
    print("CLM restart and ET at (i, j) = ({}, {})".format(I, J))
    print("=" * 60)
    print("\nNote: ET is NOT in restart files; it is in CLM binary output (qflx_evap_tot).")
    print("Restart files contain state variables only.\n")

    # 1x1 single file
    rst_24h_file = os.path.join(RST_24H_DIR, "clm.rst.00000.0")
    dummy_candidates = [
        os.path.join(DUMMY_RST_DIR, "clm.rst.00000.0"),
        rst_24h_file,
    ]

    dummy_data = None
    dummy_label = None
    for path in dummy_candidates:
        if os.path.isfile(path):
            dummy_data = read_restart_quiet(path)
            dummy_label = path
            break
    if dummy_label:
        print("--- Dummy / initial restart ---")
        print("File:", dummy_label)
        print_restart_at(dummy_data, "dummy", I, J)
        print()
    else:
        print("No dummy restart file found under", DUMMY_RST_DIR)
        print()

    if os.path.isfile(rst_24h_file):
        print("--- 24h run restart ---")
        print("File:", rst_24h_file)
        data_24h = read_restart_quiet(rst_24h_file)
        print_restart_at(data_24h, "24h", I, J)
        print()
    else:
        print("24h restart file not found:", rst_24h_file)
        print()

    # ET from CLM binary if present (1x1: nc=107, nr=89 from YAML)
    nc, nr = 107, 89
    for run_dir, ts, label in [
        (RST_24H_DIR, 24, "24h run"),
        (DUMMY_RST_DIR, 24, "dummy run (ts=24)"),
        (DUMMY_RST_DIR, 0, "dummy run (ts=0)"),
    ]:
        et = read_et_from_binary(run_dir, ts, nc, nr, I, J)
        if et is not None:
            print("ET (qflx_evap_tot) at ({}, {}): {}  [{}]  [mm/s]".format(I, J, et, label))
        else:
            print("ET binary not found for {} (ts={}) at {}".format(label, ts, run_dir))
    print()
    print("Done.")


if __name__ == "__main__":
    main()
