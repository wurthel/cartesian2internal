import pytest
from pathlib import Path
from cartesian2internal import run
from tests.utils import cwd, assert_comp_forces


@pytest.fixture
def root() -> Path:
    return Path(__file__).parent / 'data'


@pytest.mark.parametrize("fname_coordinates, fname_forces, fname_zmatrix", [
    ("traj.xyz", "forces", "zmatrix")])
def test_cartesian2internal(tmp_path, root, fname_coordinates, fname_forces, fname_zmatrix):
    with cwd(tmp_path):
        fname_coordinates = root / fname_coordinates
        fname_forces = root / fname_forces
        fname_zmatrix = root / fname_zmatrix

        out_file = run(fname_coordinates,
                       fname_forces,
                       fname_zmatrix)

        file1 = root / out_file.name
        file2 = Path(out_file)
        assert_comp_forces(file1, file2)
