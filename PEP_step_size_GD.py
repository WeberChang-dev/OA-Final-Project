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
    delta = 1e-4
    
    print(f"Running grid search over {grid_size}x{grid_size} ({grid_size**2}) stepsize pairs...")
    h1_coords = []
    h2_coords = []
    certified = []
    for i, h1 in enumerate(tqdm(h_vals)):
        for j, h2 in enumerate(h_vals):
            # Matrix coordinates: rows index h2 (Y-axis), cols index h1 (X-axis)
            is_straightforward = check_straightforward(h1, h2, L=1.0, D=1.0, delta=delta)
            h1_coords.append(h1)
            h2_coords.append(h2)
            certified.append(is_straightforward)

    h1_coords = np.array(h1_coords)
    h2_coords = np.array(h2_coords)
    certified = np.array(certified)
    
    plt.figure(figsize=(7, 6))
    
    # Plot only the certified configurations, dropping uncertified combinations completely
    plt.scatter(h1_coords[certified], h2_coords[certified], 
                color='#1f77b4', marker='o', s=45, alpha=0.9, label="Certified Straightforward")
    
    # Layout and labels
    plt.title(rf"GD Step Patterns (Certified Dots Only, $\Delta = {delta}$)", fontsize=11)
    plt.xlabel("$h_1$")
    plt.ylabel("$h_2$")
    plt.xlim(0.4, 3.2)
    plt.ylim(0.4, 3.2)
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # Reference stepsize threshold indicator lines
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.4, label="$h=1.0$ Standard Step")
    plt.axvline(1.0, color='red', linestyle='--', alpha=0.4)
    
    plt.legend(loc='upper right', framealpha=0.9)
    plt.show()               
            
    # # Plotting the resulting certified region
    # plt.figure(figsize=(7, 6))
    # plt.imshow(results, origin='lower', extent=[0.5, 3.0, 0.5, 3.0], 
    #            cmap='YlGnBu', alpha=0.85)
    
    # plt.title(f"GD Stepsize Patterns Certified as Straightforward, $\Delta = {delta}$)")
    # plt.xlabel("$h_1$")
    # plt.ylabel("$h_2$")
    # plt.grid(True, linestyle='--', alpha=0.5)
    
    # # Highlight the standard h=1.0 bound
    # plt.axhline(1.0, color='red', linestyle=':', alpha=0.6, label="$h=1.0$ Standard Stepsize")
    # plt.axvline(1.0, color='red', linestyle=':', alpha=0.6)
    # plt.legend(loc='upper right')
    
    plt.show()

if __name__ == "__main__":
    run_figure_2_grid_search()