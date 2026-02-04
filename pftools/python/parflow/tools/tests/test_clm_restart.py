"""
Unit tests for the CLM restart module.

This verifies the CLMRestartReader and CLMRestartWriter classes,
as well as the round-trip functionality (read -> write -> read -> compare).

This test can be run standalone or via ctest.
"""

import sys
import os
import numpy as np
import tempfile
from pathlib import Path

# Add parent directory to path for imports
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, rootdir)

from parflow.tools.clm_restart import (
    CLMRestartReader,
    CLMRestartWriter,
    calculate_processor_topology,
    get_rank_subdomain,
)


class TestCLMRestart:
    """Test cases for CLM restart file I/O."""

    def __init__(self):
        """Set up test fixtures."""
        self.nlevsoi = 10
        self.nlevsno = 5
        self.nlayers = self.nlevsoi + self.nlevsno
        self.errors = []

    def _create_test_restart_data(self, nch=4):
        """Create minimal test restart data for testing."""
        return {
            # Metadata
            'yr': 2024, 'mo': 1, 'da': 1,
            'hr': 0, 'mn': 0, 'ss': 0,
            'vclass': 1,
            'nc': 2, 'nr': 2, 'nch': nch,
            'istep': 100,

            # Grid indices (1-based)
            'col': np.array([1, 2, 1, 2], dtype=np.int32),
            'row': np.array([1, 1, 2, 2], dtype=np.int32),

            # Scalar fields
            'fgrd': np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64),
            'vegt': np.array([1, 2, 1, 2], dtype=np.int32),
            't_grnd': np.array([273.15, 274.15, 275.15, 276.15], dtype=np.float64),
            't_veg': np.array([273.0, 274.0, 275.0, 276.0], dtype=np.float64),
            'h2osno': np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
            'snowage': np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            'snowdp': np.array([0.0, 0.01, 0.02, 0.03], dtype=np.float64),
            'h2ocan': np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
            'frac_sno': np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
            'elai': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            'esai': np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64),
            'snl': np.array([0, 0, 0, 0], dtype=np.int32),
            'xerr': np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            'zerr': np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),

            # Layer fields
            'dz': np.ones((nch, self.nlayers), dtype=np.float64) * 0.1,
            'z': np.ones((nch, self.nlayers), dtype=np.float64) * 0.05,
            'zi': np.ones((nch, self.nlayers + 1), dtype=np.float64) * 0.1,
            't_soisno': np.ones((nch, self.nlayers), dtype=np.float64) * 273.15,
            'h2osoi_liq': np.ones((nch, self.nlayers), dtype=np.float64) * 0.5,
            'h2osoi_ice': np.zeros((nch, self.nlayers), dtype=np.float64),
        }

    def test_read_restart_file(self):
        """Test reading a CLM restart file."""
        # Create test data
        test_data = self._create_test_restart_data()

        # Write test file
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / 'test_restart.rst'
            writer = CLMRestartWriter(self.nlevsoi, self.nlevsno)
            writer.write(test_file, test_data)

            # Read it back
            reader = CLMRestartReader(self.nlevsoi, self.nlevsno)
            data = reader.read(test_file)

            # Verify metadata
            assert data['yr'] == 2024, f"Expected yr=2024, got {data['yr']}"
            assert data['mo'] == 1, f"Expected mo=1, got {data['mo']}"
            assert data['da'] == 1, f"Expected da=1, got {data['da']}"
            assert data['istep'] == 100, f"Expected istep=100, got {data['istep']}"
            assert data['nch'] == 4, f"Expected nch=4, got {data['nch']}"
            assert data['nc'] == 2, f"Expected nc=2, got {data['nc']}"
            assert data['nr'] == 2, f"Expected nr=2, got {data['nr']}"

            # Verify grid indices
            np.testing.assert_array_equal(data['col'], test_data['col'])
            np.testing.assert_array_equal(data['row'], test_data['row'])

            # Verify scalar fields
            np.testing.assert_allclose(data['t_grnd'], test_data['t_grnd'], rtol=1e-14)
            np.testing.assert_allclose(data['fgrd'], test_data['fgrd'], rtol=1e-14)
            np.testing.assert_array_equal(data['vegt'], test_data['vegt'])

            # Verify layer fields
            np.testing.assert_allclose(data['dz'], test_data['dz'], rtol=1e-14)
            np.testing.assert_allclose(data['t_soisno'], test_data['t_soisno'], rtol=1e-14)

    def test_round_trip(self):
        """Test round-trip: read -> write -> read -> compare."""
        # Create test data
        test_data = self._create_test_restart_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write original file
            original_file = Path(tmpdir) / 'original.rst'
            writer = CLMRestartWriter(self.nlevsoi, self.nlevsno)
            writer.write(original_file, test_data)

            # Read original
            reader = CLMRestartReader(self.nlevsoi, self.nlevsno)
            data1 = reader.read(original_file)

            # Write to new location
            output_file = Path(tmpdir) / 'roundtrip.rst'
            writer.write(output_file, data1)

            # Read back
            data2 = reader.read(output_file)

            # Compare metadata
            metadata_keys = ['yr', 'mo', 'da', 'hr', 'mn', 'ss', 'vclass', 'nc', 'nr', 'nch', 'istep']
            for key in metadata_keys:
                if data1[key] != data2[key]:
                    self.errors.append(f"Metadata mismatch for {key}: {data1[key]} != {data2[key]}")

            # Compare integer arrays
            int_array_keys = ['col', 'row', 'vegt', 'snl']
            for key in int_array_keys:
                try:
                    np.testing.assert_array_equal(
                        data1[key], data2[key],
                        err_msg=f"Integer array mismatch for {key}"
                    )
                except AssertionError as e:
                    self.errors.append(str(e))

            # Compare float arrays (with tolerance)
            float_keys = [
                'fgrd', 't_grnd', 't_veg', 'h2osno', 'snowage', 'snowdp',
                'h2ocan', 'frac_sno', 'elai', 'esai', 'xerr', 'zerr',
                'dz', 'z', 'zi', 't_soisno', 'h2osoi_liq', 'h2osoi_ice'
            ]

            for key in float_keys:
                try:
                    np.testing.assert_allclose(
                        data1[key], data2[key],
                        rtol=1e-14, atol=1e-14,
                        err_msg=f"Float array mismatch for {key}"
                    )
                except AssertionError as e:
                    self.errors.append(str(e))

    def test_calculate_processor_topology(self):
        """Test processor topology calculation."""
        # Test simple case
        topo = calculate_processor_topology(nx=40, ny=40, P=4, Q=4)
        assert topo['P'] == 4, f"Expected P=4, got {topo['P']}"
        assert topo['Q'] == 4, f"Expected Q=4, got {topo['Q']}"
        assert topo['nx'] == 40, f"Expected nx=40, got {topo['nx']}"
        assert topo['ny'] == 40, f"Expected ny=40, got {topo['ny']}"
        assert len(topo['nc0']) == 4, f"Expected len(nc0)=4, got {len(topo['nc0'])}"
        assert len(topo['nr0']) == 4, f"Expected len(nr0)=4, got {len(topo['nr0'])}"
        # Each processor should get 10 cells
        assert np.all(topo['nc0'] == 10), f"Expected all nc0=10, got {topo['nc0']}"
        assert np.all(topo['nr0'] == 10), f"Expected all nr0=10, got {topo['nr0']}"

        # Test with remainder
        topo = calculate_processor_topology(nx=41, ny=41, P=4, Q=4)
        # First processor should get 11, rest get 10
        assert topo['nc0'][0] == 11, f"Expected nc0[0]=11, got {topo['nc0'][0]}"
        assert topo['nc0'][1] == 10, f"Expected nc0[1]=10, got {topo['nc0'][1]}"
        assert topo['nc0'][2] == 10, f"Expected nc0[2]=10, got {topo['nc0'][2]}"
        assert topo['nc0'][3] == 10, f"Expected nc0[3]=10, got {topo['nc0'][3]}"

    def test_get_rank_subdomain(self):
        """Test getting subdomain information for a rank."""
        topo = calculate_processor_topology(nx=40, ny=40, P=4, Q=4)

        # Test rank 0 (top-left)
        subdomain = get_rank_subdomain(0, topo)
        assert subdomain['rank'] == 0, f"Expected rank=0, got {subdomain['rank']}"
        assert subdomain['conc'] == 0, f"Expected conc=0, got {subdomain['conc']}"
        assert subdomain['conr'] == 0, f"Expected conr=0, got {subdomain['conr']}"
        assert subdomain['ix_start'] == 0, f"Expected ix_start=0, got {subdomain['ix_start']}"
        assert subdomain['iy_start'] == 0, f"Expected iy_start=0, got {subdomain['iy_start']}"
        assert subdomain['nc'] == 10, f"Expected nc=10, got {subdomain['nc']}"
        assert subdomain['nr'] == 10, f"Expected nr=10, got {subdomain['nr']}"

        # Test rank 5 (conc=1, conr=1)
        subdomain = get_rank_subdomain(5, topo)
        assert subdomain['rank'] == 5, f"Expected rank=5, got {subdomain['rank']}"
        assert subdomain['conc'] == 1, f"Expected conc=1, got {subdomain['conc']}"
        assert subdomain['conr'] == 1, f"Expected conr=1, got {subdomain['conr']}"
        assert subdomain['ix_start'] == 10, f"Expected ix_start=10, got {subdomain['ix_start']}"
        assert subdomain['iy_start'] == 10, f"Expected iy_start=10, got {subdomain['iy_start']}"

    def test_read_restart_file_with_data(self):
        """Test reading a real CLM restart file if test data is available."""
        test_file = os.path.join(rootdir, "tools/tests/data/clm.rst.00000.0")
        if not os.path.exists(test_file):
            print(f"Skipping test_read_restart_file_with_data: test data not found at {test_file}")
            return

        reader = CLMRestartReader(self.nlevsoi, self.nlevsno)
        data = reader.read(test_file)

        # Basic sanity checks
        assert data['nch'] > 0, f"Expected nch > 0, got {data['nch']}"
        assert data['nc'] > 0, f"Expected nc > 0, got {data['nc']}"
        assert data['nr'] > 0, f"Expected nr > 0, got {data['nr']}"
        assert data['dz'].shape[0] == data['nch'], f"Expected dz.shape[0]={data['nch']}, got {data['dz'].shape[0]}"
        assert data['dz'].shape[1] == self.nlayers, f"Expected dz.shape[1]={self.nlayers}, got {data['dz'].shape[1]}"
        assert data['zi'].shape[1] == self.nlayers + 1, f"Expected zi.shape[1]={self.nlayers + 1}, got {data['zi'].shape[1]}"

    def test_round_trip_with_data(self):
        """Test round-trip with real CLM restart file if test data is available."""
        test_file = os.path.join(rootdir, "tools/tests/data/clm.rst.00000.0")
        if not os.path.exists(test_file):
            print(f"Skipping test_round_trip_with_data: test data not found at {test_file}")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = CLMRestartReader(self.nlevsoi, self.nlevsno)
            writer = CLMRestartWriter(self.nlevsoi, self.nlevsno)

            # Read original
            data1 = reader.read(test_file)

            # Write to temp location
            output_file = Path(tmpdir) / 'roundtrip.rst'
            writer.write(output_file, data1)

            # Read back
            data2 = reader.read(output_file)

            # Compare all fields
            metadata_keys = ['yr', 'mo', 'da', 'hr', 'mn', 'ss', 'vclass', 'nc', 'nr', 'nch', 'istep']
            for key in metadata_keys:
                if data1[key] != data2[key]:
                    self.errors.append(f"Metadata mismatch for {key}: {data1[key]} != {data2[key]}")

            int_array_keys = ['col', 'row', 'vegt', 'snl']
            for key in int_array_keys:
                try:
                    np.testing.assert_array_equal(
                        data1[key], data2[key],
                        err_msg=f"Integer array mismatch for {key}"
                    )
                except AssertionError as e:
                    self.errors.append(str(e))

            float_keys = [
                'fgrd', 't_grnd', 't_veg', 'h2osno', 'snowage', 'snowdp',
                'h2ocan', 'frac_sno', 'elai', 'esai', 'xerr', 'zerr',
                'dz', 'z', 'zi', 't_soisno', 'h2osoi_liq', 'h2osoi_ice'
            ]

            for key in float_keys:
                try:
                    np.testing.assert_allclose(
                        data1[key], data2[key],
                        rtol=1e-14, atol=1e-14,
                        err_msg=f"Float array mismatch for {key}"
                    )
                except AssertionError as e:
                    self.errors.append(str(e))


def main():
    """Run all tests and exit with appropriate code."""
    test = TestCLMRestart()
    
    print("=" * 70)
    print("Running CLM Restart Tests")
    print("=" * 70)
    
    try:
        print("\n1. Testing read_restart_file...")
        test.test_read_restart_file()
        print("   ✓ PASSED")
        
        print("\n2. Testing round_trip...")
        test.test_round_trip()
        if test.errors:
            print(f"   ✗ FAILED with {len(test.errors)} error(s)")
            for error in test.errors:
                print(f"     - {error}")
            test.errors = []  # Clear errors for next test
        else:
            print("   ✓ PASSED")
        
        print("\n3. Testing calculate_processor_topology...")
        test.test_calculate_processor_topology()
        print("   ✓ PASSED")
        
        print("\n4. Testing get_rank_subdomain...")
        test.test_get_rank_subdomain()
        print("   ✓ PASSED")
        
        print("\n5. Testing read_restart_file_with_data...")
        test.test_read_restart_file_with_data()
        print("   ✓ PASSED (or skipped if data not available)")
        
        print("\n6. Testing round_trip_with_data...")
        test.test_round_trip_with_data()
        if test.errors:
            print(f"   ✗ FAILED with {len(test.errors)} error(s)")
            for error in test.errors:
                print(f"     - {error}")
        else:
            print("   ✓ PASSED (or skipped if data not available)")
        
        print("\n" + "=" * 70)
        if test.errors:
            print("TESTS FAILED")
            print("=" * 70)
            return 1
        else:
            print("ALL TESTS PASSED")
            print("=" * 70)
            return 0
            
    except Exception as e:
        print(f"\n✗ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
