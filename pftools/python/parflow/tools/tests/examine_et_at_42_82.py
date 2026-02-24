#!/usr/bin/env python3
"""
Examine CLM restart variables and ET at grid position (i, j) = (42, 82).

Note: ET (evapotranspiration) is NOT stored in CLM restart (.rst) files.
Restart files hold state variables (t_grnd, h2osoi_liq, etc.).
ET is in CLM output PFB files: <run_name>.out.clm_output.<timestep>.C.pfb.
This script:
  - Prints all restart variables at (i,j)=(42,82) for dummy and 24h rst files.
  - Reads ET (qflx_evap_tot) from CLM output PFB at (42,82).
"""
import glob
import os
import re
import sys
import numpy as np

# Default paths under build/test/python/test_output
BASE = "/home/ga6/workspace/parflow/build/test/python/test_output"
RUN_NAME = "clm_restart_real_data"
# Dummy: initial/main run dir (may have clm.rst at start or we use 24h for both if only one exists)
DUMMY_RST_DIR = os.path.join(BASE, 'clm_restart_real_data_reference_2x2')
# 24h run: first 24h restart
RST_24H_DIR = os.path.join(BASE, f"{RUN_NAME}_restart_first24h")

# Grid position (1-based column i, row j as in ParFlow/CLM)
I, J = 42, 82

# CLM layer counts (match test)
NLEVSOI = 4
NLEVSNO = 5

# Order of variables in SingleFile CLM output PFB (must match ParFlow io.py clm_output_variables)
CLM_OUTPUT_VARIABLES = (
    "eflx_lh_tot",
    "eflx_lwrad_out",
    "eflx_sh_tot",
    "eflx_soil_grnd",
    "qflx_evap_tot",
    "qflx_evap_grnd",
    "qflx_evap_soi",
    "qflx_evap_veg",
    "qflx_tran_veg",
    "qflx_infl",
    "swe_out",
    "t_grnd",
    "qflx_qirr",
    "t_soil",
)
QFLX_EVAP_TOT_INDEX = CLM_OUTPUT_VARIABLES.index("qflx_evap_tot")


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


def find_clm_output_pfb_files(run_dir):
    """
    Find CLM output PFB files in run_dir.
    Files are named like: <run_name>.out.clm_output.<timestep>.C.pfb
    (and may have a companion .pfb.dist; we need the .pfb path for reading).
    Returns list of (filepath, timestep_int) sorted by timestep.
    """
    # Match *.out.clm_output.NNNNN.C.pfb (exclude .pfb.dist)
    pattern = os.path.join(run_dir, "*.out.clm_output.*.C.pfb")
    candidates = glob.glob(pattern)
    out = []
    for path in candidates:
        if path.endswith(".dist"):
            continue
        base = os.path.basename(path)
        # e.g. restart_run.out.clm_output.00024.C.pfb
        m = re.search(r"\.out\.clm_output\.(\d+)\.C\.pfb$", base)
        if m:
            ts = int(m.group(1))
            out.append((path, ts))
    out.sort(key=lambda x: x[1])
    return out


def read_et_from_clm_output_pfb(pfb_path, i, j):
    """
    Read qflx_evap_tot at (i, j) from a SingleFile CLM output PFB.
    PFB shape is (nz, ny, nx) with z_first=True; i,j are 1-based column/row.
    qflx_evap_tot is at layer index QFLX_EVAP_TOT_INDEX.
    """
    from parflow.tools.io import read_pfb
    arr = read_pfb(pfb_path, z_first=True)
    if arr.ndim != 3:
        return None
    nz, ny, nx = arr.shape
    if QFLX_EVAP_TOT_INDEX >= nz:
        return None
    # (i, j) 1-based -> array indices (j-1, i-1)
    return float(arr[QFLX_EVAP_TOT_INDEX, j - 1, i - 1])


def main():
    print("=" * 60)
    print("CLM restart and ET at (i, j) = ({}, {})".format(I, J))
    print("=" * 60)
    print("\nNote: ET is NOT in restart files; it is in CLM output PFB files")
    print("(<run_name>.out.clm_output.<timestep>.C.pfb). Restart files contain state only.\n")

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

    # ET from CLM output PFB files
    print("--- ET (qflx_evap_tot) from CLM output PFB ---")
    for run_dir, label in [
        (RST_24H_DIR, "24h run"),
        (DUMMY_RST_DIR, "dummy/main run"),
    ]:
        files = find_clm_output_pfb_files(run_dir)
        if not files:
            print("No CLM output PFB files found in {} [{}]".format(run_dir, label))
            continue
        print("{}: {} file(s) in {}".format(label, len(files), run_dir))
        for pfb_path, ts in files:
            et = read_et_from_clm_output_pfb(pfb_path, I, J)
            if et is not None:
                print("  ts={:05d}: qflx_evap_tot at ({}, {}) = {}  [mm/s]".format(ts, I, J, et))
            else:
                print("  ts={:05d}: could not read ET from {}".format(ts, os.path.basename(pfb_path)))
    print()
    print("Done.")


if __name__ == "__main__":
    main()
