"""
CS 124
Programming Assignment #2
Kelechi Ukah and Thomas Maldonado
"""
import sys
import random

def random_mat(n, seq = (0,1)):
    '''
    Create a random n by n matrix filled with elements from seq iterable
    '''
    A = [[random.choice(seq) for i in range(n)] for j in range(n)]
    return A

def print_mat(A):
    '''
    Helper function to print matrix
    '''
    for row in A:
        print(' '.join(map(str, row)))
    print()

def pad(A):
    '''
    add zeros column and row
    '''
    for row in A:
        row.append(0)
    A.append([0 for i in range(len(A[0]))])

    return A

def unpad(A):
    '''
    remove a column of zeros and ones
    '''
    for row in A:
        row.pop()
    A.pop()

    return A

def split(M):
    '''
    Helpler function to divide the submatrices
    '''
    n = len(M) // 2

    A, B, C, D = [[[0 for i in range(n)] for i in range(n)] for i in range(4)]
    for i in range(n):
        for j in range(n):
            A[i][j] = M[i][j]
            B[i][j] = M[i][j + n]
            C[i][j] = M[i + n][j]
            D[i][j] = M[i + n][j + n]

    return A, B, C, D

def add(A, B, sign = 1):
    '''
    add (or subtract) two matrices
    '''
    n = len(A)
    C = [[0 for i in range(n)] for j in range(n)]

    if sign:
        for i in range(n):
            for j in range(n):
                C[i][j] = A[i][j] + B[i][j]
    else:
        for i in range(n):
            for j in range(n):
                C[i][j] = A[i][j] - B[i][j]

    return C

def merge(A, B, C, D):
    n = len(A)
    M = [[0 for i in range(2*n)] for j in range(2*n)]
    for i in range(n):
        for j in range(n):
            M[i][j] = A[i][j]
            M[i][j + n] = B[i][j]
            M[i + n][j] = C[i][j]
            M[i + n][j + n] = D[i][j]

    return M

def mat_mul(A, B):
    '''
    Conventional multiplication of two n by n matrices: A, B
    '''
    # initialize dimension of matrix and number of operations
    n = len(A)
    # initialize final matrix
    C = [[0 for i in range(n)] for j in range(n)]
    # implement contraction
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k]*B[k][j]
    return C

def strassen(M1, M2, n0 = 0):
    n = len(M1)
    # base case
    if n == 1:
        return [[M1[0][0]*M2[0][0]]]

    if n <= n0:
      return mat_mul(M1, M2)

    # if n is odd pad and unpad later before recursion
    if n % 2:
        M1 = pad(M1)
        M2 = pad(M2)

    # split into quadrants
    A, B, C, D = split(M1)
    E, F, G, H = split(M2)

    P1 = strassen(A, add(F, H, 0), n0)
    P2 = strassen(add(A, B), H, n0)
    P3 = strassen(add(C, D), E, n0)
    P4 = strassen(D, add(G, E, 0), n0)
    P5 = strassen(add(A, D), add(E, H), n0)
    P6 = strassen(add(B, D, 0), add(G, H), n0)
    P7 = strassen(add(A, C, 0), add(E, F), n0)

    # calculate strass submatrices
    AEBG = add(add(add(P5, P4), P2, 0), P6)
    AFBH = add(P1, P2)
    CEDG = add(P3, P4)
    CFDH = add(add(add(P5, P1), P3, 0), P7, 0)

    M = merge(AEBG, AFBH, CEDG, CFDH)

    # if n is odd
    if n % 2:
        M = unpad(M)
        M1 = unpad(M1)
        M2 = unpad(M2)

    return M

if __name__ == "__main__":
    flag = sys.argv[1]
    dimension = int(sys.argv[2])
    inputfile = sys.argv[3]
    file = open(inputfile, 'r')

    # Initialize input matrices A and B from inputfile
    A, B = [[[0 for i in range(dimension)] for j in range(dimension)] for k in range(2)]
    for i in range(dimension ** 2):
      row = i // dimension
      col = i - row * dimension
      A[row][col] = int(file.readline())
    for i in range(dimension ** 2):
      row = i // dimension
      col = i - row * dimension
      B[row][col] = int(file.readline())

    # Perform Strassen's algorithm on input matrices, print diagonal entries
    n0 = 405
    C = strassen(A, B, n0)
    for i in range(dimension):
      print(C[i][i])
