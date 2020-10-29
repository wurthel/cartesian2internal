import numpy as np
from pathlib import Path
from transformation import transform
import sys


def run(fname_coordinates, fname_forces, fname_zmatrix):
    coordinates = []
    with open(fname_coordinates, 'r') as f:
        line = f.readline()
        while line:
            num_atoms = int(line)
            line = f.readline()
            line = f.readline()
            step = []
            for _ in range(num_atoms):
                _, *xs = line.split()
                xs = [float(x) for x in xs]
                step.append(xs)
                line = f.readline()
            coordinates.append(step)
    coordinates = np.array(coordinates)

    forces = []
    with open(fname_forces, 'r') as f:
        line = f.readline()
        while line:
            while line and "# Atom   Kind   Element" not in f.readline():
                True
            line = f.readline()
            step = []
            while line and "SUM OF ATOMIC FORCES" not in line:
                _, _, _, *xs = line.split()
                xs = [float(x) for x in xs]
                step.append(xs)
                line = f.readline()
            line = f.readline()
            forces.append(step)
    forces = np.array(forces)

    zmatrix = []
    with open(fname_zmatrix, 'r') as f:
        for line in f.readlines():
            type, ids = line.split()
            ids = [int(x) - 1 for x in ids.split(',')]
            zmatrix.append((type, ids))

    out_file = Path(fname_forces.stem + "-out" + fname_forces.suffix)
    with open(out_file, 'w') as f:
        for i, (x, dEdx) in enumerate(zip(coordinates, forces)):
            new_forces = transform(x, dEdx, zmatrix)

            line = f"i = {i}\n"
            for ic, force in zip(zmatrix, new_forces):
                line += f"{ic[0]:<8}{force}\n"
            line += "\n"
            f.write(line)

    return out_file


if __name__ == "__main__":
    fname_coordinates = Path(sys.argv[1])
    fname_forces = Path(sys.argv[2])
    fname_zmatrix = Path(sys.argv[3])
    run(fname_coordinates, fname_forces, fname_zmatrix)
