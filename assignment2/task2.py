import numpy as np
A = np.array([[2, 3, 1], [4, 11, -1], [-2, 1, 7]], dtype=float)
b = np.array([10, 33, 15], dtype=float)
def gaussian_elimination_with_pivoting(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)]) #Augmenting the Matrix, It converts the 1D array b into a column vector

    for i in range(n):    # Forward elimination with pivoting
        pivot_row = np.argmax(abs(Ab[i:, i])) + i #find the row with the largest element in the current column
        if i != pivot_row:  # Swap the current row with the pivot row
            Ab[[i, pivot_row]] = Ab[[pivot_row, i]]
        for j in range(i + 1, n): # Eliminate entries below the pivot
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
    print("Upper triangular matrix [A|b]:")
    print(Ab)

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1): #solve for x starting from the last row
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:])) / Ab[i, i]
    print("Solution:")
    print(x)
    return x
solution = gaussian_elimination_with_pivoting(A, b)
