import numpy as np
# The coefficient matrix and right-hand side vector
A = np.array([[10, -1, -2], [-2, 10, -1], [-1, -2, 10]], dtype=float)
b = np.array([5, -6, 15], dtype=float)
x0 = np.array([0, 0, 0], dtype=float) # Initial guess
tol = 1e-6
max_iter = 100
# Checking for Diagonal Dominance
def is_diagonally_dominant(A):
    for i in range(len(A)): #if for each row
        row_sum = sum(abs(A[i, j]) for j in range(len(A)) if j != i)
        if abs(A[i, i]) <= row_sum: #the abs value of DE >= sum of the abs values
            return False
    return True
def jacobi_method(A, b, x0, tol, max_iter):
    n = len(b)
    x = x0.copy() #Copies initial guess
    x_new = np.zeros_like(x) #sets up placeholder for updated values
    print(f"Initial guess: {x0}")
    print("-" * 40)

    for it in range(max_iter):
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]  # Jacobi Formula
        print(f"Iteration {it + 1}: {x_new}")

        # Checking for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, it + 1
        x = x_new.copy()
    raise ValueError("Jacobi method did not converge within the maximum number of iterations")

is_dominant = is_diagonally_dominant(A)
# Output whether diagonal dominance criterion is satisfied
if is_dominant:
    print("Matrix is diagonally dominant. Jacobi method is expected to converge.")
else:
    print("Matrix is not diagonally dominant. Convergence is not guaranteed.")
try: #Usage and solving the system
    solution, iterations = jacobi_method(A, b, x0, tol, max_iter)
    print(f"Solution: {solution}")
    print(f"Number of iterations: {iterations}")
except ValueError as e: print(e)











