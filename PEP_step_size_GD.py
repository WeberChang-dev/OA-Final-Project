import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def check_straightforward(h1, h2, L=1.0, D=1.0, delta=1e-4):
    """
    Solves the PEP SDP for a given pair of stepsizes (h1, h2).
    Returns True if the pattern is certified as straightforward.
    """
    # 4 Basis Vectors: v0 = x0 - x*, v1 = g0, v2 = g1, v3 = g2
    G = cp.Variable((4, 4), PSD=True)
    
    # Function values at iterates ['star', 0, 1, 2]
    # We fix f['star'] = 0 without loss of generality
    f = {
        'star': 0.0,
        0: cp.Variable(),
        1: cp.Variable(),
        2: cp.Variable()
    }
    
    # Coordinate representations of (x_i - x_*) in our basis
    coeff_x = {
        'star': np.array([0.0, 0.0, 0.0, 0.0]),
        0:      np.array([1.0, 0.0, 0.0, 0.0]),
        1:      np.array([1.0, -h1/L, 0.0, 0.0]),
        2:      np.array([1.0, -h1/L, -h2/L, 0.0])
    }
    
    # Coordinate representations of gradients g_i in our basis
    coeff_g = {
        'star': np.array([0.0, 0.0, 0.0, 0.0]),
        0:      np.array([0.0, 1.0, 0.0, 0.0]),
        1:      np.array([0.0, 0.0, 1.0, 0.0]),
        2:      np.array([0.0, 0.0, 0.0, 1.0])
    }
    
    # Base constraints
    constraints = [
        f[0] <= delta,         # Initial gap constraint
        G[0, 0] <= D**2       # Initial distance constraint ||x0 - x*||^2 <= D^2
    ]
    
    # Generate N*(N-1) smooth convex interpolation conditions
    iterates = ['star', 0, 1, 2]
    for i in iterates:
        for j in iterates:
            if i == j:
                continue
            cx_diff = coeff_x[i] - coeff_x[j]
            cg_diff = coeff_g[i] - coeff_g[j]
            
            # f_i >= f_j + <g_j, x_i - x_j> + 1/(2L) * ||g_i - g_j||^2
            linear_term = coeff_g[j] @ G @ cx_diff
            quad_term = (1.0 / (2.0 * L)) * cp.quad_form(cg_diff, G)
            
            constraints.append(f[i] >= f[j] + linear_term + quad_term)
            
    # Maximize the final objective gap after 2 steps
    obj = cp.Maximize(f[2])
    prob = cp.Problem(obj, constraints)
    
    try:
        # Using SCS as a reliable, bundled first-order solver fallback
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-6)
        
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # Straightforwardness safety threshold check
            threshold = delta - ((h1 + h2) / (L * D**2)) * (delta**2)
            # Returns True if worst case behaves better than or equal to the formula
            return prob.value <= threshold + 1e-7
        return False
    except Exception:
        return False

def run_figure_2_grid_search():
    # Setup grid search matching Figure 2 parameters
    h_vals = np.arange(0.5, 3.1, 0.1)
    grid_size = len(h_vals)
    results = np.zeros((grid_size, grid_size))
    
    print(f"Running grid search over {grid_size}x{grid_size} ({grid_size**2}) stepsize pairs...")
    
    for i, h1 in enumerate(tqdm(h_vals)):
        for j, h2 in enumerate(h_vals):
            # Matrix coordinates: rows index h2 (Y-axis), cols index h1 (X-axis)
            is_straightforward = check_straightforward(h1, h2, L=1.0, D=1.0, delta=1e-4)
            results[j, i] = 1.0 if is_straightforward else 0.0                 
            
    # Plotting the resulting certified region
    plt.figure(figsize=(7, 6))
    plt.imshow(results, origin='lower', extent=[0.5, 3.0, 0.5, 3.0], 
               cmap='Blues', alpha=0.7)
    
    plt.title("GD Stepsize Patterns Certified as Straightforward (Figure 2)")
    plt.xlabel("$h_1$")
    plt.ylabel("$h_2$")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Highlight specific notable points mentioned in the paper (e.g., h = (2.9, 1.5))
    if 2.9 in h_vals and 1.5 in h_vals:
        plt.scatter([2.9], [1.5], color='red', marker='*', s=150, label="Stable long-step (2.9, 1.5)")
        plt.legend()
        
    plt.show()

if __name__ == "__main__":
    run_figure_2_grid_search()