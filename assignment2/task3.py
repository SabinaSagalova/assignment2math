import numpy as np
A = np.array([[1, 1, 1], [2, -3, 4], [3, 4, 5]], dtype=float)
b = np.array([9, 13, 40], dtype=float)
def gauss_jordan(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)]) # Augment matrix A with vector b
    for i in range(n): # Transform into reduced row-echelon form (diagonal form)
        Ab[i] = Ab[i] / Ab[i, i] #normalize the pivot row
        for j in range(n): #this eliminate Other Entries in the Column
            if i != j:
                factor = Ab[j, i]
                Ab[j] -= factor * Ab[i]
    print("Diagonal matrix [A|b]:")
    print(Ab)
    x = Ab[:, -1] # Extract the solution, which is last column
    print("Solution:")
    print(x)
    return x
solution = gauss_jordan(A, b)

