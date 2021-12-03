import numpy as np


def qr(A):
    """
    Finds the QR decomposition of a given matrix A
    :param A: np.array
    :return: Q (np.array), R (np.array)
    """
    orth_set = gram_schmidt(A)
    Q = create_Q(orth_set, A)
    R = create_R(orth_set, A)
    return Q, R


def create_R(orth_set, A):
    """
    Builds the R matrix of the QR decomposition of a given matrix A
    :param orth_set: list[] of np.array's: the columns of matrix A after apply the G-S process on them
    :param A: np.array
    :return: R (np.array)
    """
    R = np.zeros((A.shape[1], A.shape[1]))
    for i in range(A.shape[1]):  # Apply numbers to diag
        R[i, i] = np.linalg.norm(orth_set[i])
    for i in range(A.shape[1]):  # Apply numbers to upper triangle
        for j in range(i + 1, A.shape[1]):
            R[i, j] = np.inner(A[:, j], orth_set[i] / np.linalg.norm(orth_set[i]))
    return R


def create_Q(orth_set, A):
    """
    Builds the Q matrix of the QR decomposition of a given matrix A
    :param orth_set: list[] of np.array's: the columns of matrix A after apply the G-S process on them
    :param A: np.array
    :return: Q (np.array)
    """
    Q = np.zeros((A.shape[0], A.shape[1]))
    for i in range(len(orth_set)):  # Fill columns with orthonormal vectors
        Q[:, i] = orth_set[i] / np.linalg.norm(orth_set[i])
    return Q


def gram_schmidt(A):
    """
    Applies the Gram-Schmidt process on the columns of a given matrix A
    :param A: np.array: A's columns will be used to find an orthogonal basis
    :return: list[] of np.array's forming an orthogonal basis
    """
    orth_set = []
    orth_set.append(A[:, 0])
    for i in range(1, A.shape[1]):
        sum = 0
        for j in range(i):
            sum += (np.inner(A[:, i], orth_set[j]) / np.inner(orth_set[j], orth_set[j])) * (orth_set[j])
        orth_set.append(A[:, i] - sum)
    return orth_set


def main():
    A = np.array([[3, 6, 8, 0, 4, 3, 1, 5, 4, 4],
                  [4, 0, 6, 5, 1, 9, 3, 3, 3, 3],
                  [5, 0, 9, 8, 0, 4, 9, 6, 6, 4],
                  [0, 7, 6, 9, 2, 5, 5, 5, 3, 4],
                  [2, 3, 8, 1, 2, 2, 6, 6, 6, 4],
                  [5, 4, 1, 8, 1, 5, 8, 9, 5, 3],
                  [0, 1, 7, 5, 3, 7, 9, 4, 0, 7],
                  [2, 9, 2, 8, 3, 4, 8, 2, 2, 5],
                  [6, 6, 0, 0, 4, 6, 8, 2, 7, 1],
                  [4, 7, 8, 6, 4, 8, 7, 8, 2, 7],
                  [7, 5, 9, 9, 5, 1, 8, 4, 3, 8],
                  [2, 4, 9, 2, 9, 4, 0, 7, 0, 8],
                  [2, 8, 2, 4, 2, 4, 6, 3, 5, 1],
                  [2, 9, 6, 8, 2, 5, 9, 0, 0, 9],
                  [1, 4, 5, 2, 2, 2, 2, 6, 9, 5]])

    x = np.array([21, 11, 9, 6, 5, 4, 2, 1, 94, 91, 89, 85, 84, 16, 98])

    Q, R = qr(A)
    np.set_printoptions(precision=2)
    print(f"Q: \n{Q}\n")
    print(f"R: \n{R}\n")
    x_proj_onto_S = np.matmul(Q, np.matmul(np.transpose(Q), x))
    print(f"Projection of x={x} onto S: \n{x_proj_onto_S}\n")
    x_proj_onto_s_orthcomp = np.matmul(np.eye(A.shape[0]) - np.matmul(Q, np.transpose(Q)), x)
    print(f"Projection of x={x} onto the orthogonal complement of S: \n{x_proj_onto_s_orthcomp}")


if __name__ == "__main__":
    main()
