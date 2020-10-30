import numpy as np


def cr(i, j):
    if i == j:
        return 1
    return 0


def sf(i, j, k):
    return cr(i, j) - cr(i, k)


def eij(xi, xj):
    e = xj - xi
    return e / np.linalg.norm(e)


def rij(xi, xj):
    return np.linalg.norm(xi - xj)


def transform(x, dEdx, zmatrix):
    in_dims = len(zmatrix)
    out_dims = len(x.reshape(-1))

    B = np.zeros((in_dims, out_dims))

    for t, (tp, mnop) in enumerate(zmatrix):
        # Bond lengths
        if 'R' in tp:
            n, m = mnop
            u = eij(x[n], x[m])

            for a in mnop:
                for i in [0, 1, 2]:
                    B[t, 3 * a + i] = sf(a, m, n) * u[i]

        # Valence Angle Bending
        if 'A' in tp:
            n, o, m = mnop
            u = eij(x[o], x[m])
            v = eij(x[o], x[n])
            ru = rij(x[o], x[m])
            rv = rij(x[o], x[n])

            if np.isclose(abs(np.dot(u, v)), 1):
                if np.isclose(abs(np.dot(u, [1, -1, 1])), 1) and np.isclose(abs(np.dot(v, [1, -1, 1])), 1):
                    w = np.cross(u, [-1, 1, 1])
                else:
                    w = np.cross(u, [1, -1, 1])
            else:
                w = np.cross(u, v)
            w = w / np.linalg.norm(w)

            uw = np.cross(u, w)
            wv = np.cross(w, v)

            for a in mnop:
                for i in [0, 1, 2]:
                    value = sf(a, m, o) * uw[i] / ru + sf(a, n, o) * wv[i] / rv
                    B[t, 3 * a + i] = value

        # Torsion
        if 'D' in tp:
            n, p, o, m = mnop
            u = eij(x[o], x[m])
            w = eij(x[o], x[p])
            v = eij(x[p], x[n])
            ru = rij(x[o], x[m])
            rw = rij(x[o], x[p])
            rv = rij(x[p], x[n])

            cosu = np.dot(u, w)
            cosv = np.dot(v, -w)
            sinu = np.linalg.norm(np.cross(u, w))
            sinv = np.linalg.norm(np.cross(v, -w))
            sinu2 = sinu * sinu
            sinv2 = sinv * sinv
            uw = np.cross(u, w)
            vw = np.cross(v, w)

            for a in mnop:
                for i in [0, 1, 2]:
                    value = sf(a, m, o) * uw[i] / (ru * sinu2) + \
                            sf(a, p, n) * vw[i] / (rv * sinv2) + \
                            sf(a, o, p) * (uw[i] * cosu / (rw * sinu2) + vw[i] * cosv / (rw * sinv2))
                    # or last term: sf(a, o, p) * (uw[i] * cosu / (rw * sinu2) - vw[i] * cosv / (rw * sinv2))
                    B[t, 3 * a + i] = value

    u = np.identity(out_dims)
    G = np.einsum("ii,ni,mi->nm", u, B, B)
    G_inv = np.linalg.pinv(G)
    B_inv = np.linalg.pinv(B)
    P = G.dot(G_inv)

    dxdy = B_inv.T
    dEdy = dxdy.dot(dEdx.reshape(-1))

    return P.dot(dEdy)
