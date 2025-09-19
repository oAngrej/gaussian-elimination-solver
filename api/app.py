import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from fractions import Fraction

# --- Backend Setup ---
app = Flask(__name__)
CORS(app)

# --- Helper Function for Fraction Formatting ---
def format_fraction(f):
    """Formats a Fraction object into a neat string."""
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"

# --- Core Logic Adapted to Use Fractions ---

def swap_rows(M, row_index_1, row_index_2):
    M = M.copy()
    M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
    return M

def get_index_first_non_zero_value_from_column(M, column, starting_row):
    column_array = M[starting_row:, column]
    for i, val in enumerate(column_array):
        if val != Fraction(0):
            return i + starting_row
    return -1

def augmented_matrix(A, B):
    return np.hstack((A, B))

def row_echelon_form(A, B):
    # Convert all inputs to Fraction objects for precise calculations
    A_frac = np.array([[Fraction(x) for x in row] for row in A], dtype=object)
    B_frac = np.array([[Fraction(x)] for x in B], dtype=object)

    # Use a dummy float matrix just for the determinant check
    A_float = A.copy().astype('float64')
    if np.isclose(np.linalg.det(A_float), 0):
        return None, [{"description": "Error: The coefficient matrix is singular (determinant is zero). The system does not have a unique solution.", "matrix": None, "isError": True}]

    M = augmented_matrix(A_frac, B_frac)
    num_rows = len(A)
    
    # Format initial matrix for logging
    matrix_str = [[format_fraction(cell) for cell in row] for row in M]
    steps = [{"description": "Starting with the Augmented Matrix [A|B].", "matrix": matrix_str}]

    for i in range(num_rows):
        pivot_candidate = M[i, i]
        
        if pivot_candidate == Fraction(0):
            swap_row_index = get_index_first_non_zero_value_from_column(M, i, i + 1)
            if swap_row_index == -1:
                 matrix_str = [[format_fraction(cell) for cell in row] for row in M]
                 return None, [{"description": "Error: Cannot find a non-zero pivot. The system does not have a unique solution.", "matrix": matrix_str, "isError": True}]
            
            M = swap_rows(M, i, swap_row_index)
            matrix_str = [[format_fraction(cell) for cell in row] for row in M]
            steps.append({"description": f"Pivot at R{i+1}C{i+1} is zero. Swapping R{i+1} with R{swap_row_index+1}.", "matrix": matrix_str})
        
        pivot = M[i, i]
        if pivot != Fraction(1):
            M[i] = M[i] / pivot
            matrix_str = [[format_fraction(cell) for cell in row] for row in M]
            steps.append({"description": f"Normalize R{i+1} to make the pivot 1. R{i+1} → R{i+1} / ({format_fraction(pivot)})", "matrix": matrix_str})
        
        for j in range(i + 1, num_rows):
            factor = M[j, i]
            if factor != Fraction(0):
                M[j] = M[j] - factor * M[i]
                matrix_str = [[format_fraction(cell) for cell in row] for row in M]
                steps.append({"description": f"Eliminate the entry at R{j+1}C{i+1}. R{j+1} → R{j+1} - ({format_fraction(factor)}) * R{i+1}", "matrix": matrix_str})

    matrix_str = [[format_fraction(cell) for cell in row] for row in M]
    steps.append({"description": "Matrix is now in Row Echelon Form. Starting Back Substitution.", "matrix": matrix_str})
    return M, steps

def back_substitution(M_ref):
    M = M_ref.copy()
    num_rows = M.shape[0]
    steps = []

    for i in range(num_rows - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            factor = M[j, i]
            if factor != Fraction(0):
                M[j] = M[j] - factor * M[i]
                matrix_str = [[format_fraction(cell) for cell in row] for row in M]
                steps.append({"description": f"Eliminate the entry at R{j+1}C{i+1}. R{j+1} → R{j+1} - ({format_fraction(factor)}) * R{i+1}", "matrix": matrix_str})
    
    solution = M[:, -1]
    matrix_str = [[format_fraction(cell) for cell in row] for row in M]
    steps.append({"description": "Reduced Row Echelon Form is achieved. The last column is the solution.", "matrix": matrix_str})
    return solution, steps

def string_to_augmented_matrix(equations_str):
    equations = [eq.strip() for eq in equations_str.strip().split('\n')]
    num_equations = len(equations)
    variables = sorted(list(set(re.findall(r'[a-zA-Z]+\d*|[a-zA-Z]', equations_str))), key=lambda var: (var.isalpha(), var))
    var_map = {var: i for i, var in enumerate(variables)}
    num_vars = len(variables)

    if num_equations != num_vars:
        return None, None, None, f"Error: Found {num_equations} equations but {num_vars} variables. The system must be square (n x n)."

    A = np.zeros((num_equations, num_vars))
    B = np.zeros((num_equations, 1))

    for i, eq in enumerate(equations):
        parts = eq.split('=')
        if len(parts) != 2: continue
        try:
            rhs = float(parts[1].strip())
            B[i] = rhs
        except ValueError:
            return None, None, None, f"Invalid number on the right side of equation {i+1}."
        
        lhs = parts[0]
        # Regex to find terms like '3*x', '-x', '+ 2.5*y', handles terms without explicit coeff
        pattern = r'([+-]?)?\s*(\d+\.?\d*|)\s*\*?\s*([a-zA-Z]+\d*|[a-zA-Z])'
        
        processed_lhs = lhs.replace(" ", "")
        if processed_lhs.startswith('-') or processed_lhs.startswith('+'):
            pass
        else:
            processed_lhs = '+' + processed_lhs

        terms = re.findall(pattern, processed_lhs)

        for sign, coeff, var in terms:
            if var in var_map:
                idx = var_map[var]
                val = float(coeff) if coeff else 1.0
                if sign == '-': val = -val
                A[i, idx] += val
    return A, B, variables, None

# --- API Endpoint ---
@app.route('/solve', methods=['POST'])
def solve():
    data = request.get_json()
    
    if data['inputType'] == 'matrix':
        try:
            A = np.array(data['A'], dtype=float)
            B = np.array(data['B'], dtype=float).flatten()
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid matrix data. Please enter numbers only."}), 400
    else:
        A, B, variables, error = string_to_augmented_matrix(data['equations'])
        if error:
            return jsonify({"error": error}), 400
        data['variableNames'] = variables
        B = B.flatten()

    row_echelon_matrix, steps1 = row_echelon_form(A, B)
    
    if row_echelon_matrix is None:
        return jsonify({"steps": steps1})

    solution_fractions, steps2 = back_substitution(row_echelon_matrix)
    
    final_solution = {name: format_fraction(val) for name, val in zip(data['variableNames'], solution_fractions)}
    
    return jsonify({
        "steps": steps1 + steps2,
        "solution": final_solution
    })



