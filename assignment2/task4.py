import numpy as np
A = np.array([[8, -3, 2], [4, 11, -1], [6, 3, 12]], dtype=float)
b = np.array([20, 33, 36], dtype=float)
x0 = np.array([0, 0, 0], dtype=float) # Initial guess
tol = 1e-5 # Tolerance and maximum iterations
max_iter = 100
def gauss_seidel(A, b, x0, tol, max_iter):
    n = len(b)
    x = x0.copy()
    table = [] #will store iteration number and solution vector
    for k in range(max_iter):
        x_old = x.copy() #holds the solution from the previous iteration
        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i))  # Sum for previous values
            s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))  # Sum for old values
            x[i] = (b[i] - s1 - s2) / A[i, i] #Formula, updates based on the current approximation
        table.append((k + 1, *x)) # Append table

        # Check for convergence and stopping criterion
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, table
    raise ValueError("Gauss-Seidel method did not converge within the maximum number of iterations")
solution, iteration_table = gauss_seidel(A, b, x0, tol, max_iter)

print("Iteration Table:")
print(f"{'Iteration':<10}{'x1':<15}{'x2':<15}{'x3':<15}")
for row in iteration_table:
    print(f"{row[0]:<10}{row[1]:<15.8f}{row[2]:<15.8f}{row[3]:<15.8f}")

print("\nSolution:")
print(f"x1 = {solution[0]:.8f}, x2 = {solution[1]:.8f}, x3 = {solution[2]:.8f}")

