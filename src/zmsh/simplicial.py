import functools
import numpy as np


def bits_set(z, size=8):
    r"""Return a list of the bits of `z` that are set"""
    return [b for b in range(size) if z & (1 << b)]


def bit_count(n):
    return bin(n).count("1")


@functools.lru_cache(maxsize=10)
def simplex_to_chain_complex(dimension: int):
    r"""Return the equivalent chain complex for a standard simplex"""
    Z = list(range(2 ** (dimension + 1)))
    X = [[i for i in Z if bit_count(i) == d] for d in range(1, dimension + 2)]
    As = [np.ones((1, dimension + 1), dtype=np.int8)]
    for faces, cells in zip(X[:-1], X[1:]):
        num_rows, num_cols = len(faces), len(cells)
        A = np.zeros((num_rows, num_cols), dtype=np.int8)
        # TODO: for the love of Christ make this less repulsive
        for col, cell in enumerate(cells):
            cell_bits = bits_set(cell)
            for row, face in enumerate(faces):
                face_bits = bits_set(face)
                if set(face_bits).issubset(set(cell_bits)):
                    removed_vertex = (set(cell_bits) - set(face_bits)).pop()
                    idx = cell_bits.index(removed_vertex)
                    A[row, col] = (-1) ** idx

        As.append(A)

    return As


def compute_matrix_transformation(A: np.ndarray, B: np.ndarray):
    r"""Return the permutation `p` and an array of signs `s` such that
    `A[:, p] @ diag(s) == B`"""
    if A.shape != B.shape:
        raise ValueError("Matrices must be the same shape!")

    p = np.zeros(A.shape[1], dtype=int)
    s = np.zeros_like(p)
    # TODO: Make all this less repulsive
    for j in range(A.shape[1]):
        a_j = A[:, j]
        matches = [
            k
            for k in range(B.shape[1])
            if np.array_equal(a_j.nonzero()[0], B[:, k].nonzero()[0])
        ]
        if len(matches) != 1:
            # TODO: More informative error message or better description
            raise ValueError("Must be only one match per column!")

        k = matches[0]
        b_k = B[:, k]
        if not (np.array_equal(a_j, b_k) or np.array_equal(a_j, -b_k)):
            # TODO what does that word even mean
            raise ValueError("Matching columns incommensurate!")

        p[j] = k
        s[j] = np.sign(np.inner(a_j, b_k))

    return p, s


def permutation_matrix(p):
    I = np.eye(len(p), dtype=int)
    return I[:, p]


def compute_morphism(As, Bs):
    if not len(As) == len(Bs):
        raise ValueError("Chain complexes must have the same dimension!")

    num_vertices = As[0].shape[0]
    ps = [np.arange(num_vertices, dtype=int)]
    ss = [np.ones(num_vertices, dtype=int)]

    for A, B in zip(As, Bs):
        P = permutation_matrix(ps[-1])
        S = np.diag(ss[-1])
        p, s = compute_matrix_transformation(A, S @ P.T @ B)
        ps.append(p)
        ss.append(s)

    return ps[1:], ss[1:]


def orientation(As):
    r"""Return +1 if the chain complex represents the positive orientation of
    the standard simplex, or -1 for the negative orientation"""
    Bs = simplex_to_chain_complex(len(As) - 1)
    ps, ss = compute_morphism(As[1:], Bs[1:])
    return ss[-1][0]


class SimplicialTopology:
    def __init__(self, dimension: int, num_cells: List[int]):
        self._adjacency = [None] + [
            np.ma.masked_array(
                np.zeros((num_cells[n], n + 1), dtype=np.int32),
                np.ones((num_cells[n], n + 1), dtype=bool),
            )
            for n in range(1, dimension + 1)
        ]

        self._coadjacency = [None] + [
            np.ma.masked_array(
                np.zeros((num_cells[n], 2), dtype=np.int32),
                np.ones((num_cells[n], 2), dtype=bool),
            )
            for n in range(1, dimension + 1)
        ]

        self._free_cells = [None] + [
            set(range(num_cells[n])) for n in range(1, dimension + 1)
        ]

    def find(self, vertices: List[int]) -> int:
        r"""Return the integer ID of the given simplex if it exists, or
        `None` if it does not"""
        dimension = len(vertices) - 1
        coadjacency = self._coadjacency[dimension]
        cells = np.unique(coadjacency[vertices, :]).compressed()
        adjacency = self._adjacency[dimension]
        for cell in cells:
            cell_vertices = adjacency[cell, :]
            if all(v in cell_vertices for v in vertices):
                return cell

        return None

    def _pop_free_cell_index(self, dimension: int) -> int:
        if len(self._free_cells[dimension]) == 0:
            raise NotImplementedError("Can't increase topology size yet!")
        return self._free_cells[dimension].pop()

    def _add1(self, dimension, vertices):
        if self.find(vertices) is not None:
            return
        index = self._pop_free_cell_index(dimension)
        self._adjacency[dimension][index] = vertices

        for vertex in vertices:
            cells = self._coadjacency[dimension][vertex, :].compressed()
            # TODO: Miracle occurs...

    def add(self, vertices):
        for dimension in range(len(vertices) - 1, 0, -1):
            for fvertices in itertools.combinations(vertices, dimension + 1):
                self._add1(dimension, fvertices)
