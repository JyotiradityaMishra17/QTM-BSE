from pyelpa import DistributedMatrix
import numpy as np
from qtm.mpi.comm import MPI4PY_INSTALLED
if MPI4PY_INSTALLED:
    from mpi4py import MPI

def parse_eigval(num_eigval, na):
    if isinstance(num_eigval, str):
        if num_eigval.lower() == "all":
            return na
        else:
            raise ValueError(
                "String input must be 'all' to indicate all eigenvalues."
            )
    elif isinstance(num_eigval, int):
        if num_eigval < 1:
            raise ValueError("The number of eigenvalues must be at least 1.")
        if num_eigval > na:
            raise ValueError(
                "The number of eigenvalues cannot exceed the matrix dimension."
            )
        return num_eigval
    else:
        raise TypeError(
            "The number of eigenvalues must be an integer or the string 'all'."
        )


def distributed_to_regular(D):
    global_mat = np.zeros((D.na, D.na), dtype=D.data.dtype)
    for local_row in range(D.na_rows):
        for local_col in range(D.na_cols):
            global_row, global_col = D.get_global_index(local_row, local_col)
            global_mat[global_row, global_col] = D.data[local_row, local_col]
    full_matrix = np.zeros_like(global_mat)
    D.processor_layout.comm.Allreduce(global_mat, full_matrix, op=MPI.SUM)
    return full_matrix


def diag_elpa(mtx: np.ndarray, num_eigval="all", size_block=8):
    na = len(mtx)
    nev = parse_eigval(num_eigval, na)
    nblk = size_block

    elpamat = DistributedMatrix.from_comm_world(na, nev, nblk, dtype=complex)

    for global_row, global_col, row_block_size, col_block_size in \
            elpamat.global_block_indices():
        block = mtx[
            global_row : global_row + row_block_size,
            global_col : global_col + col_block_size,
        ]
        elpamat.set_block_for_global_index(
            global_row, global_col, row_block_size, col_block_size, block
        )


    data = elpamat.compute_eigenvectors()
    eigenvalues = data["eigenvalues"]
    eigenvectors = data["eigenvectors"]
    eigenvectors = distributed_to_regular(eigenvectors)
    return eigenvalues, eigenvectors