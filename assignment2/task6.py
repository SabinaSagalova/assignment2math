import numpy as np
def ill_conditioned_example():
    A = np.array([[1.001, 0.999], [1.002, 1.000]], dtype=float)
    b = np.array([2.0, 2.001], dtype=float)

    # Analytical solution using Cramer's rule
    det_A = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0] #Computes the determinant of A
    x1_analytical = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det_A #Formula
    x2_analytical = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det_A
    analytical_solution = np.array([x1_analytical, x2_analytical])
    numerical_solution = np.linalg.solve(A, b) # Numerical solution using numpy's solver

    # Defining the perturbed system, Small changes maded to matrix
    A_mod = np.array([[1.001, 0.998], [1.003, 1.001]], dtype=float)
    b_mod = np.array([2.0, 2.002], dtype=float)
    numerical_solution_mod = np.linalg.solve(A_mod, b_mod)

    print("Original System:")
    print("Analytical Solution:", analytical_solution)
    print("Numerical Solution:", numerical_solution)

    print("\nModified System:")
    print("Numerical Solution with Modified Coefficients:", numerical_solution_mod)
ill_conditioned_example()
