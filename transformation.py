import numpy as np
import itertools


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
    D = np.zeros((in_dims, out_dims, out_dims))

    for t, (tp, mnop) in enumerate(zmatrix):
        # Bond lengths
        if 'R' in tp :
            n, m = mnop
            u = eij(x[n], x[m])
            r = rij(x[n], x[m])

            for a in mnop:
                for i in [0, 1, 2]:
                    B[t, 3 * a + i] = sf(a, m, n) * u[i]

            for a, b in itertools.product(mnop, mnop):
                for i, j in itertools.product([0, 1, 2], [0, 1, 2]):
                    D[t, 3 * a + i, 3 * b + j] = ((-1) ** cr(a, b)) * (u[i] * u[j] - cr(i, j)) / r

        # Valence Angle Bending
        if 'A' in tp:
            n, o, m = mnop
            u = eij(x[o], x[m])
            v = eij(x[o], x[n])
            ru = rij(x[o], x[m])
            rv = rij(x[o], x[n])
            cos = np.dot(u, v)
            sin = np.linalg.norm(np.cross(u, v))

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

            for a, b in itertools.product(mnop, mnop):
                for i, j in itertools.product([0, 1, 2], [0, 1, 2]):
                    value = sf(a, m, o) * sf(b, m, o) * (
                            u[i] * v[j] + u[j] * v[i] - 3 * u[i] * u[j] * cos + cr(i, j) * cos) / (
                                    ru * ru * sin)
                    value += sf(a, n, o) * sf(b, n, o) * (
                            v[i] * u[j] + v[j] * u[i] - 3 * v[i] * v[j] * cos + cr(i, j) * cos) / (
                                     rv * rv * sin)
                    value += sf(a, m, o) * sf(b, n, o) * (
                            u[i] * u[j] + v[j] * v[i] - u[i] * v[j] * cos - cr(i, j)) / (
                                     ru * rv * sin)
                    value += sf(a, n, o) * sf(b, m, o) * (
                            v[i] * v[j] + u[j] * u[i] - v[i] * u[j] * cos - cr(i, j)) / (
                                     ru * rv * sin)
                    value += (-1) * cos * B[t, 3 * a + i] * B[t, 3 * b + j] / sin
                    D[t, 3 * b + j, 3 * a + i] = value

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
            cosu2 = cosu * cosu
            cosv2 = cosv * cosv
            uw = np.cross(u, w)
            vw = np.cross(v, w)

            for a in mnop:
                for i in [0, 1, 2]:
                    value = sf(a, m, o) * uw[i] / (ru * sinu2) + \
                            sf(a, p, n) * vw[i] / (rv * sinv2) + \
                            sf(a, o, p) * (uw[i] * cosu / (rw * sinu2) + vw[i] * cosv / (rw * sinv2))
                    # or last term: sf(a, o, p) * (uw[i] * cosu / (rw * sinu2) - vw[i] * cosv / (rw * sinv2))
                    B[t, 3 * a + i] = value

            for a, b in itertools.product(mnop, mnop):
                for i, j in itertools.product([0, 1, 2], [0, 1, 2]):
                    value = sf(a, m, o) * sf(b, m, o) * (
                            uw[i] * (w[j] * cosu - u[j]) + uw[j] * (w[i] * cosu - u[i])) / (
                                    ru * ru * sinu2 * sinu2)
                    value += sf(a, n, p) * sf(b, n, p) * (
                            vw[i] * (w[j] * cosv - v[j]) + vw[j] * (w[i] * cosv - v[i])) / (
                                     rv * rv * sinv2 * sinv2)
                    value += (sf(a, m, o) * sf(b, o, p) + sf(a, p, o) * sf(b, o, m)) * (
                            uw[i] * (w[j] - 2 * u[j] * cosu + w[j] * cosu2) +
                            uw[j] * (w[i] - 2 * u[i] * cosu + w[i] * cosu2)) / (2 * ru * rw * sinu2 * sinu2)
                    value += (sf(a, n, p) * sf(b, p, o) + sf(a, p, o) * sf(b, n, p)) * (
                            vw[i] * (w[j] + 2 * u[j] * cosv + w[j] * cosv2) +
                            vw[j] * (w[i] + 2 * u[i] * cosv + w[i] * cosv2)) / (2 * rv * rw * sinv2 * sinv2)
                    value += sf(a, o, p) * sf(b, p, o) * (
                            uw[i] * (u[j] + u[j] * cosu2 - 3 * w[j] * cosu + w[j] * cosu2 * cosu) +
                            uw[j] * (u[i] + u[i] * cosu2 - 3 * w[i] * cosu + w[i] * cosu2 * cosu)) / (
                                     2 * rw * rw * sinu2 * sinu2)
                    value += sf(a, o, p) * sf(b, o, p) * (
                            vw[i] * (v[j] + v[j] * cosv2 + 3 * w[j] * cosv - w[j] * cosv2 * cosv) +
                            vw[j] * (v[i] + v[i] * cosv2 + 3 * w[i] * cosv - w[i] * cosv2 * cosv)) / (
                                     2 * rw * rw * sinv2 * sinv2)
                    if i != j:
                        k = {0, 1, 2}.difference({i, j}).pop()
                        value += (1 - cr(a, b)) * (sf(a, m, o) * sf(b, o, p) + sf(a, p, o) * sf(b, o, m)) * (
                                j - i) * (-0.5) ** abs(j - i) * (w[k] * cosu - u[k]) / (ru * rw * sinu)
                        value += (1 - cr(a, b)) * (sf(a, n, o) * sf(b, o, p) + sf(a, p, o) * sf(b, o, m)) * (
                                j - i) * (-0.5) ** abs(j - i) * (w[k] * cosv - v[k]) / (rv * rw * sinv)

                    D[t, 3 * b + j, 3 * a + i] = value

    u = np.identity(out_dims)
    G = np.einsum("ii,ni,mi->nm", u, B, B)
    G_inv = np.linalg.pinv(G)
    B_inv = np.linalg.pinv(B)
    P = G.dot(G_inv)

    dxdy = B_inv.T
    dEdy = dxdy.dot(dEdx.reshape(-1))

    return P.dot(dEdy)
