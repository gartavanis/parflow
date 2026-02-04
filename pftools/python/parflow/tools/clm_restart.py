# -*- coding: utf-8 -*-
"""clm_restart module

CLM Restart File Redistribution Tool

This module provides utilities to redistribute CLM restart files when
changing processor topology (P*Q) between ParFlow-CLM simulation runs.

Author: Reed Maxwell (with Claude Code assistance)
Date: 2025-12-28
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import struct


# ============================================================================
# Binary File I/O Classes
# ============================================================================

class CLMRestartReader:
    """
    Read CLM restart files in Fortran unformatted binary format.

    CLM restart files are written by Fortran code using unformatted I/O,
    which includes record length markers before and after each record.

    Format:
        - Little-endian byte order (native on most systems)
        - Record structure: [4-byte length][data][4-byte length]
        - Integers: 4 bytes (int32)
        - Reals: 8 bytes (float64)
    """

    def __init__(self, nlevsoi: int = 10, nlevsno: int = 5):
        """
        Initialize reader.

        Args:
            nlevsoi: Number of soil layers (default: 10)
            nlevsno: Number of snow layers (default: 5)
        """
        self.nlevsoi = nlevsoi
        self.nlevsno = nlevsno
        self.nlayers = nlevsoi + nlevsno

    def _read_fortran_record(self, f, dtype: str, count: int) -> np.ndarray:
        """
        Read a Fortran unformatted record.

        Fortran unformatted format:
            [4 bytes: record length]
            [data: count * size(dtype)]
            [4 bytes: record length (repeated)]

        Args:
            f: Open file handle
            dtype: Data type ('i' for int32, 'd' for float64)
            count: Number of elements to read

        Returns:
            NumPy array of data
        """
        # Read record length header (4 bytes, little-endian int32)
        rec_len_bytes = f.read(4)
        if len(rec_len_bytes) != 4:
            raise EOFError("Unexpected end of file reading record length")
        rec_len = struct.unpack('<i', rec_len_bytes)[0]

        # Read data based on type
        if dtype == 'i':
            # 32-bit integers
            data = np.fromfile(f, dtype='<i4', count=count)
            expected_size = count * 4
        elif dtype == 'd':
            # 64-bit floats (doubles)
            data = np.fromfile(f, dtype='<f8', count=count)
            expected_size = count * 8
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

        # Verify we read the right amount
        if len(data) != count:
            raise ValueError(f"Expected {count} values, got {len(data)}")

        # Read record length footer (should match header)
        rec_len_check = struct.unpack('<i', f.read(4))[0]
        if rec_len != rec_len_check:
            raise ValueError(
                f"Fortran record length mismatch: "
                f"header={rec_len}, footer={rec_len_check}"
            )

        # Verify expected size
        if rec_len != expected_size:
            raise ValueError(
                f"Record length mismatch: "
                f"expected={expected_size}, got={rec_len}"
            )

        return data

    def read(self, filepath: Path) -> Dict[str, Any]:
        """
        Read a CLM restart file.

        Args:
            filepath: Path to restart file

        Returns:
            Dictionary containing all restart data
        """
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            # Read header (10 integers)
            header = self._read_fortran_record(f, 'i', 10)
            yr, mo, da, hr, mn, ss, vclass, nc, nr, nch = header

            print(f"  Header: {yr}-{mo:02d}-{da:02d} {hr:02d}:{mn:02d}:{ss:02d}")
            print(f"  Grid: nc={nc}, nr={nr}, nch={nch}")

            # Read grid indices (1-based in Fortran)
            col = self._read_fortran_record(f, 'i', nch)
            row = self._read_fortran_record(f, 'i', nch)

            # Read scalar fields (floats)
            fgrd = self._read_fortran_record(f, 'd', nch)
            vegt = self._read_fortran_record(f, 'i', nch)
            t_grnd = self._read_fortran_record(f, 'd', nch)
            t_veg = self._read_fortran_record(f, 'd', nch)
            h2osno = self._read_fortran_record(f, 'd', nch)
            snowage = self._read_fortran_record(f, 'd', nch)
            snowdp = self._read_fortran_record(f, 'd', nch)
            h2ocan = self._read_fortran_record(f, 'd', nch)
            frac_sno = self._read_fortran_record(f, 'd', nch)
            elai = self._read_fortran_record(f, 'd', nch)
            esai = self._read_fortran_record(f, 'd', nch)
            snl = self._read_fortran_record(f, 'i', nch)
            xerr = self._read_fortran_record(f, 'd', nch)
            zerr = self._read_fortran_record(f, 'd', nch)

            # Read timestep
            istep_arr = self._read_fortran_record(f, 'i', 1)
            istep = int(istep_arr[0])

            # Read layer data
            # Layers go from -nlevsno+1 to nlevsoi (e.g., -4 to 10 for 15 layers)
            nlayers_dz = self.nlevsoi + self.nlevsno  # e.g., 15
            nlayers_zi = self.nlevsoi + self.nlevsno + 1  # e.g., 16 (one more for interface)

            # dz: layer depth
            dz = np.zeros((nch, nlayers_dz))
            for l in range(nlayers_dz):
                dz[:, l] = self._read_fortran_record(f, 'd', nch)

            # z: layer thickness
            z = np.zeros((nch, nlayers_dz))
            for l in range(nlayers_dz):
                z[:, l] = self._read_fortran_record(f, 'd', nch)

            # zi: interface level
            zi = np.zeros((nch, nlayers_zi))
            for l in range(nlayers_zi):
                zi[:, l] = self._read_fortran_record(f, 'd', nch)

            # t_soisno: soil + snow temperature
            t_soisno = np.zeros((nch, nlayers_dz))
            for l in range(nlayers_dz):
                t_soisno[:, l] = self._read_fortran_record(f, 'd', nch)

            # h2osoi_liq: liquid water content
            h2osoi_liq = np.zeros((nch, nlayers_dz))
            for l in range(nlayers_dz):
                h2osoi_liq[:, l] = self._read_fortran_record(f, 'd', nch)

            # h2osoi_ice: ice content
            h2osoi_ice = np.zeros((nch, nlayers_dz))
            for l in range(nlayers_dz):
                h2osoi_ice[:, l] = self._read_fortran_record(f, 'd', nch)

        return {
            # Metadata
            'yr': int(yr), 'mo': int(mo), 'da': int(da),
            'hr': int(hr), 'mn': int(mn), 'ss': int(ss),
            'vclass': int(vclass),
            'nc': int(nc), 'nr': int(nr), 'nch': int(nch),
            'istep': istep,

            # Grid indices
            'col': col.astype(np.int32),
            'row': row.astype(np.int32),

            # Scalar fields
            'fgrd': fgrd,
            'vegt': vegt.astype(np.int32),
            't_grnd': t_grnd,
            't_veg': t_veg,
            'h2osno': h2osno,
            'snowage': snowage,
            'snowdp': snowdp,
            'h2ocan': h2ocan,
            'frac_sno': frac_sno,
            'elai': elai,
            'esai': esai,
            'snl': snl.astype(np.int32),
            'xerr': xerr,
            'zerr': zerr,

            # Layer fields
            'dz': dz,
            'z': z,
            'zi': zi,
            't_soisno': t_soisno,
            'h2osoi_liq': h2osoi_liq,
            'h2osoi_ice': h2osoi_ice,
        }


class CLMRestartWriter:
    """
    Write CLM restart files in Fortran unformatted binary format.
    """

    def __init__(self, nlevsoi: int = 10, nlevsno: int = 5):
        """
        Initialize writer.

        Args:
            nlevsoi: Number of soil layers (default: 10)
            nlevsno: Number of snow layers (default: 5)
        """
        self.nlevsoi = nlevsoi
        self.nlevsno = nlevsno
        self.nlayers = nlevsoi + nlevsno

    def _write_fortran_record(self, f, data: np.ndarray, dtype: str):
        """
        Write a Fortran unformatted record.

        Args:
            f: Open file handle
            data: Data to write (will be converted to appropriate dtype)
            dtype: Data type ('i' for int32, 'd' for float64)
        """
        # Convert data to bytes
        if dtype == 'i':
            data_array = np.asarray(data, dtype='<i4')
        elif dtype == 'd':
            data_array = np.asarray(data, dtype='<f8')
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

        data_bytes = data_array.tobytes()
        rec_len = len(data_bytes)

        # Write: [length][data][length]
        f.write(struct.pack('<i', rec_len))
        f.write(data_bytes)
        f.write(struct.pack('<i', rec_len))

    def write(self, filepath: Path, data: Dict[str, Any]):
        """
        Write a CLM restart file.

        Args:
            filepath: Path for output file
            data: Dictionary with all restart data
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            # Header (10 integers)
            header = np.array([
                data['yr'], data['mo'], data['da'],
                data['hr'], data['mn'], data['ss'],
                data['vclass'],
                data['nc'], data['nr'], data['nch']
            ], dtype=np.int32)
            self._write_fortran_record(f, header, 'i')

            # Grid indices
            self._write_fortran_record(f, data['col'], 'i')
            self._write_fortran_record(f, data['row'], 'i')

            # Scalar fields
            self._write_fortran_record(f, data['fgrd'], 'd')
            self._write_fortran_record(f, data['vegt'], 'i')
            self._write_fortran_record(f, data['t_grnd'], 'd')
            self._write_fortran_record(f, data['t_veg'], 'd')
            self._write_fortran_record(f, data['h2osno'], 'd')
            self._write_fortran_record(f, data['snowage'], 'd')
            self._write_fortran_record(f, data['snowdp'], 'd')
            self._write_fortran_record(f, data['h2ocan'], 'd')
            self._write_fortran_record(f, data['frac_sno'], 'd')
            self._write_fortran_record(f, data['elai'], 'd')
            self._write_fortran_record(f, data['esai'], 'd')
            self._write_fortran_record(f, data['snl'], 'i')
            self._write_fortran_record(f, data['xerr'], 'd')
            self._write_fortran_record(f, data['zerr'], 'd')

            # Timestep
            self._write_fortran_record(f, np.array([data['istep']]), 'i')

            # Layer data
            nlayers_dz = self.nlevsoi + self.nlevsno
            nlayers_zi = nlayers_dz + 1

            # dz
            for l in range(nlayers_dz):
                self._write_fortran_record(f, data['dz'][:, l], 'd')

            # z
            for l in range(nlayers_dz):
                self._write_fortran_record(f, data['z'][:, l], 'd')

            # zi
            for l in range(nlayers_zi):
                self._write_fortran_record(f, data['zi'][:, l], 'd')

            # t_soisno
            for l in range(nlayers_dz):
                self._write_fortran_record(f, data['t_soisno'][:, l], 'd')

            # h2osoi_liq
            for l in range(nlayers_dz):
                self._write_fortran_record(f, data['h2osoi_liq'][:, l], 'd')

            # h2osoi_ice
            for l in range(nlayers_dz):
                self._write_fortran_record(f, data['h2osoi_ice'][:, l], 'd')


# ============================================================================
# Topology Calculation
# ============================================================================

def calculate_processor_topology(nx: int, ny: int, P: int, Q: int) -> Dict[str, Any]:
    """
    Calculate processor topology using ParFlow's distribution algorithm.

    Args:
        nx, ny: Domain dimensions
        P, Q: Number of processors in each direction

    Returns:
        Dictionary with topology information
    """
    # Base subgrid size (floor division)
    mg_nx = nx // P
    mg_ny = ny // Q

    # Remainder cells
    rm_nx = nx % P
    rm_ny = ny % Q

    # Distribution: first rm_nx processors get mg_nx+1 cells, rest get mg_nx
    nc0 = np.array([mg_nx + 1 if p < rm_nx else mg_nx for p in range(P)])
    nr0 = np.array([mg_ny + 1 if q < rm_ny else mg_ny for q in range(Q)])

    return {
        'P': P, 'Q': Q,
        'nx': nx, 'ny': ny,
        'nc0': nc0, 'nr0': nr0,
        'mg_nx': mg_nx, 'mg_ny': mg_ny,
        'rm_nx': rm_nx, 'rm_ny': rm_ny
    }


def get_rank_subdomain(rank: int, topology: Dict[str, Any]) -> Dict[str, int]:
    """
    Get subdomain information for a specific rank.

    Args:
        rank: Processor rank (0-based)
        topology: Topology dictionary from calculate_processor_topology

    Returns:
        Dictionary with subdomain information
    """
    P, Q = topology['P'], topology['Q']
    nc0, nr0 = topology['nc0'], topology['nr0']
    mg_nx, mg_ny = topology['mg_nx'], topology['mg_ny']
    rm_nx, rm_ny = topology['rm_nx'], topology['rm_ny']

    # Processor grid position (matches Fortran: conc = mod(rank,P)+1, but 0-based)
    conc = rank % P
    conr = rank // P

    # Subdomain size
    nc = nc0[conc]
    nr = nr0[conr]
    nch = nc * nr

    # Starting indices
    ix_start = conc * mg_nx + min(conc, rm_nx)
    iy_start = conr * mg_ny + min(conr, rm_ny)

    return {
        'rank': rank,
        'conc': conc,
        'conr': conr,
        'nc': int(nc),
        'nr': int(nr),
        'nch': int(nch),
        'ix_start': ix_start,
        'iy_start': iy_start,
        'ix_end': ix_start + nc,
        'iy_end': iy_start + nr
    }


# ============================================================================
# Main Redistribution Function
# ============================================================================

def redistribute_clm_restart(
        nx: int, ny: int,
        old_P: int, old_Q: int,
        new_P: int, new_Q: int,
        old_restart_dir: str,
        new_restart_dir: str,
        restart_prefix: str = 'clm.rst.',
        tstamp: int = 0,
        nlevsoi: int = 10,
        nlevsno: int = 5,
        col_row_file: Optional[Tuple[str, str]] = None):
    """
    Redistribute CLM restart files from old to new processor topology.

    This function reads CLM restart files from an old processor topology
    and redistributes them to a new topology, enabling continuation of
    ParFlow-CLM simulations with different parallel configurations.

    Args:
        nx, ny: Domain dimensions
        old_P, old_Q: Old processor topology
        new_P, new_Q: New processor topology
        old_restart_dir: Directory with old restart files
        new_restart_dir: Directory for new restart files
        restart_prefix: Restart file prefix (default: 'clm.rst.')
        tstamp: Timestamp for restart files (default: 0)
        nlevsoi: Number of soil layers (default: 10)
        nlevsno: Number of snow layers (default: 5)
        col_row_file: Optional tuple of (col_file, row_file) paths containing
                     tile ordering from ParFlow. If None, uses sequential ordering.

    Example:
        >>> from parflow.tools import redistribute_clm_restart
        >>> redistribute_clm_restart(
        ...     nx=41, ny=41,
        ...     old_P=4, old_Q=4,
        ...     new_P=6, new_Q=6,
        ...     old_restart_dir='./old_restart',
        ...     new_restart_dir='./new_restart',
        ...     col_row_file=('col_new.txt', 'row_new.txt')
        ... )
    """
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback if tqdm not available
        def tqdm(iterable, desc=None):
            for item in iterable:
                yield item

    print(f"\n{'='*70}")
    print(f"CLM Restart Redistribution")
    print(f"{'='*70}")
    print(f"Domain: {nx}×{ny}")
    print(f"Old topology: {old_P}×{old_Q} = {old_P*old_Q} ranks")
    print(f"New topology: {new_P}×{new_Q} = {new_P*new_Q} ranks")
    print(f"Restart prefix: {restart_prefix}")

    # Calculate topologies
    old_topo = calculate_processor_topology(nx, ny, old_P, old_Q)
    new_topo = calculate_processor_topology(nx, ny, new_P, new_Q)

    print(f"\nOld distribution (X): {list(old_topo['nc0'])}")
    print(f"Old distribution (Y): {list(old_topo['nr0'])}")
    print(f"New distribution (X): {list(new_topo['nc0'])}")
    print(f"New distribution (Y): {list(new_topo['nr0'])}")

    # Initialize reader/writer
    reader = CLMRestartReader(nlevsoi, nlevsno)
    writer = CLMRestartWriter(nlevsoi, nlevsno)

    # Create output directory
    new_restart_dir = Path(new_restart_dir)
    new_restart_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Assemble global domain from old files
    print(f"\n{'='*70}")
    print("Step 1: Reading old restart files and assembling global domain")
    print(f"{'='*70}")

    global_data = _assemble_global_domain(
        old_topo, Path(old_restart_dir), restart_prefix, tstamp, reader
    )

    # Step 2: Write new distribution
    print(f"\n{'='*70}")
    print("Step 2: Writing new restart files")
    print(f"{'='*70}")

    _write_new_distribution(
        new_topo, new_restart_dir, restart_prefix, tstamp, global_data, writer,
        col_row_file=col_row_file
    )

    print(f"\n{'='*70}")
    print(f"✓ Successfully redistributed CLM restart files!")
    print(f"  {old_P}×{old_Q} → {new_P}×{new_Q}")
    print(f"  Output: {new_restart_dir}")
    print(f"{'='*70}\n")


def _assemble_global_domain(topology, restart_dir, prefix, tstamp, reader):
    """Assemble global domain from distributed restart files."""
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, desc=None):
            total = len(iterable) if hasattr(iterable, '__len__') else None
            for i, item in enumerate(iterable):
                if total and desc:
                    print(f"\r{desc}: {i+1}/{total}", end='', flush=True)
                yield item
            if desc:
                print()  # newline after progress

    nx, ny = topology['nx'], topology['ny']
    nranks = topology['P'] * topology['Q']

    # Initialize global arrays
    global_data = {'metadata': None, 'scalar': {}, 'layer': {}}

    scalar_fields = ['vegt', 'fgrd', 't_grnd', 't_veg', 'h2osno',
                    'snowage', 'snowdp', 'h2ocan', 'frac_sno',
                    'elai', 'esai', 'snl', 'xerr', 'zerr']

    for field in scalar_fields:
        global_data['scalar'][field] = np.zeros((nx, ny))

    nlayers = reader.nlevsoi + reader.nlevsno
    layer_fields = ['dz', 'z', 't_soisno', 'h2osoi_liq', 'h2osoi_ice']

    for field in layer_fields:
        global_data['layer'][field] = np.zeros((nx, ny, nlayers))

    # zi has one extra layer
    global_data['layer']['zi'] = np.zeros((nx, ny, nlayers + 1))

    # Read each rank's file
    for rank in tqdm(range(nranks), desc="Reading ranks"):
        filepath = restart_dir / f"{prefix}{tstamp:05d}.{rank}"

        if not filepath.exists():
            raise FileNotFoundError(f"Restart file not found: {filepath}")

        data = reader.read(filepath)

        # Save metadata from first file
        if global_data['metadata'] is None:
            global_data['metadata'] = {
                'yr': data['yr'], 'mo': data['mo'], 'da': data['da'],
                'hr': data['hr'], 'mn': data['mn'], 'ss': data['ss'],
                'vclass': data['vclass'], 'istep': data['istep']
            }

        # Get subdomain info
        subdomain = get_rank_subdomain(rank, topology)

        # Map tiles to global grid
        for i in range(data['nch']):
            col_local = data['col'][i] - 1  # Convert to 0-based
            row_local = data['row'][i] - 1

            col_global = subdomain['ix_start'] + col_local
            row_global = subdomain['iy_start'] + row_local

            # Scalar fields
            for field in scalar_fields:
                global_data['scalar'][field][col_global, row_global] = data[field][i]

            # Layer fields
            for field in layer_fields:
                global_data['layer'][field][col_global, row_global, :] = data[field][i, :]

            # zi has different size
            global_data['layer']['zi'][col_global, row_global, :] = data['zi'][i, :]

    return global_data


def _write_new_distribution(topology, restart_dir, prefix, tstamp, global_data, writer,
                            col_row_file=None):
    """Write new distribution from global arrays.

    Args:
        topology: New topology dictionary
        restart_dir: Output directory for restart files
        prefix: Restart file prefix
        tstamp: Timestamp for restart files
        global_data: Assembled global domain data
        writer: CLMRestartWriter instance
        col_row_file: Optional tuple of (col_file_path, row_file_path) containing
                     tile ordering information. If provided, uses tile ordering from
                     these files (as generated by ParFlow). If None, uses sequential
                     col-major ordering.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, desc=None):
            total = len(iterable) if hasattr(iterable, '__len__') else None
            for i, item in enumerate(iterable):
                if total and desc:
                    print(f"\r{desc}: {i+1}/{total}", end='', flush=True)
                yield item
            if desc:
                print()  # newline after progress

    nranks = topology['P'] * topology['Q']

    # Read col/row ordering from files if provided
    col_row_data = None
    if col_row_file is not None:
        col_file, row_file = col_row_file
        print(f"Reading tile ordering from {col_file} and {row_file}...")

        col_row_data = []
        with open(col_file, 'r') as f:
            for line in f:
                col_row_data.append([int(x) for x in line.split()])

        row_data = []
        with open(row_file, 'r') as f:
            for line in f:
                row_data.append([int(x) for x in line.split()])

        if len(col_row_data) != nranks or len(row_data) != nranks:
            raise ValueError(f"Col/row files have {len(col_row_data)}/{len(row_data)} ranks, "
                           f"expected {nranks}")

        # Combine into list of (col, row) tuples for each rank
        col_row_data = [(col_row_data[r], row_data[r]) for r in range(nranks)]

    for rank in tqdm(range(nranks), desc="Writing ranks"):
        filepath = restart_dir / f"{prefix}{tstamp:05d}.{rank}"

        subdomain = get_rank_subdomain(rank, topology)
        nc, nr = subdomain['nc'], subdomain['nr']
        nch = nc * nr

        # Get col/row indices from file or generate sequentially
        if col_row_data is not None:
            col_indices = np.array(col_row_data[rank][0][:nch], dtype=np.int32)
            row_indices = np.array(col_row_data[rank][1][:nch], dtype=np.int32)
        else:
            # Generate sequential col-major indices
            col_indices = np.zeros(nch, dtype=np.int32)
            row_indices = np.zeros(nch, dtype=np.int32)
            k = 0
            for j in range(nr):
                for i in range(nc):
                    col_indices[k] = i + 1
                    row_indices[k] = j + 1
                    k += 1

        # Extract subdomain data
        restart_data = {
            **global_data['metadata'],
            'nc': nc, 'nr': nr, 'nch': nch,
            'col': col_indices,
            'row': row_indices
        }

        # Initialize scalar fields
        for field in global_data['scalar'].keys():
            restart_data[field] = np.zeros(nch)

        # Initialize layer fields
        nlayers = writer.nlevsoi + writer.nlevsno
        for field in global_data['layer'].keys():
            if field == 'zi':
                restart_data[field] = np.zeros((nch, nlayers + 1))
            else:
                restart_data[field] = np.zeros((nch, nlayers))

        # Fill from global arrays using col/row indices
        for k in range(nch):
            # col/row are 1-based local subdomain indices
            col_local = col_indices[k] - 1  # Convert to 0-based
            row_local = row_indices[k] - 1

            # Convert to global indices
            col_global = subdomain['ix_start'] + col_local
            row_global = subdomain['iy_start'] + row_local

            # Scalar fields
            for field in global_data['scalar'].keys():
                restart_data[field][k] = \
                    global_data['scalar'][field][col_global, row_global]

            # Layer fields
            for field in global_data['layer'].keys():
                restart_data[field][k, :] = \
                    global_data['layer'][field][col_global, row_global, :]

        # Write file
        writer.write(filepath, restart_data)
