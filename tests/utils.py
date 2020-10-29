import os
from contextlib import contextmanager


@contextmanager
def cwd(path):
    old_pwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_pwd)


def assert_comp_forces(file1, file2):
    forces1 = []
    with open(file1, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line or 'i' in line:
                continue
            force = float(line.split()[1])
            forces1.append(force)

    forces2 = []
    with open(file2, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line or 'i' in line:
                continue
            force = float(line.split()[1])
            forces1.append(force)

    for f1, f2 in zip(forces1, forces2):
        assert f1 == f2
