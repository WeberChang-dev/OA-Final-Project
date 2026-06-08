import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def check_nolips_straightforward(h1, h2, L=1.0, D=1.0, delta=1e-4):
    """
    Solves the PEP SDP for the NoLips (Bregman GD) method over a 2-step cycle.
    Certifies if the pattern is straightforward under relative smoothness.
    """
    # 10x10 Gram Matrix for inner products.
    # We explicitly track points (x), gradients (g), and mirror gradients (s = grad phi)
    G = cp.Variable((10, 10), PSD=True)
    
    # Function values for f(x) and hat_f(x) = L*phi(x) - f(x)
    f = cp.Variable(4)
    hat_f = cp.Variable(4)
    
    # Index mapping for the function value arrays
    v_idx = {'star': 3, 0: 0, 1: 1, 2: 2}
    
    def ip(v1, i1, v2, i2):
        """Helper to fetch <v1_i1, v2_i2> from the Gram matrix, treating x* and g* as 0."""
        if i1 == 'star' and v1 in ['x', 'g']: return 0.0
        if i2 == 'star' and v2 in ['x', 'g']: return 0.0
        
        idx_x = {0: 0, 1: 1, 2: 2}
        idx_g = {0: 3, 1: 4, 2: 5}
        idx_s = {0: 6, 1: 7, 2: 8, 'star': 9}
        
        row = idx_x[i1] if v1 == 'x' else (idx_g[i1] if v1 == 'g' else idx_s[i1])
        col = idx_x[i2] if v2 == 'x' else (idx_g[i2] if v2 == 'g' else idx_s[i2])
        return G[row, col]

    constraints = [
        # Baseline anchoring
        f[v_idx['star']] == 0.0,
        hat_f[v_idx['star']] == 0.0,
        
        # Initial objective gap
        f[v_idx[0]] <= delta,
        
        # Initial Bregman distance bound: D_phi(x*, x0) <= D^2
        # Since L*phi = hat_f + f, we can express the Bregman distance entirely 
        # using our tracked function values and inner products without defining phi directly.
        -hat_f[v_idx[0]] - f[v_idx[0]] + L * ip('s', 0, 'x', 0) <= L * D**2
    ]
    
    # Algorithmic updates (Mirror Steps in the dual space)
    # s_{k+1} = s_k - (h_k / L) * g_k
    # Applying this linearly across the Gram matrix rows
    constraints.append(G[7, :] == G[6, :] - (h1 / L) * G[3, :])
    constraints.append(G[8, :] == G[7, :] - (h2 / L) * G[4, :])
    
    # Interpolation conditions for L-Relative Smoothness
    iterates = ['star', 0, 1, 2]
    for i in iterates:
        for j in iterates:
            if i == j:
                continue
                
            # 1. f(x) must be convex: f_i >= f_j + <g_j, x_i - x_j>
            ip_g_x = ip('g', j, 'x', i) - ip('g', j, 'x', j)
            constraints.append(f[v_idx[i]] >= f[v_idx[j]] + ip_g_x)
            
            # 2. hat_f(x) must be convex: hat_f_i >= hat_f_j + <hat_g_j, x_i - x_j>
            # where hat_g_j = L*s_j - g_j
            ip_hatg_x = (L * ip('s', j, 'x', i) - ip('g', j, 'x', i)) - \
                        (L * ip('s', j, 'x', j) - ip('g', j, 'x', j))
            constraints.append(hat_f[v_idx[i]] >= hat_f[v_idx[j]] + ip_hatg_x)

    # Maximize the final objective gap f(x_2) - f(x_*)
    obj = cp.Maximize(f[v_idx[2]])
    prob = cp.Problem(obj, constraints)
    
    try:
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-5)
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # The straightforwardness formula
            threshold = delta - ((h1 + h2) / (L * D**2)) * (delta**2)
            # Check against threshold with a slight numerical buffer
            return prob.value <= threshold + 1e-6
        return False
    except Exception:
        return False

def run_nolips_grid_search():
    # Grid search matching the resolution of Figure 2
    h_vals = np.arange(0.5, 3.0, 0.1)
    grid_size = len(h_vals)
    delta = 1e-7
    h1_coords = []
    h2_coords = []
    certified = []
    
    print(f"Running NoLips PEP Grid Search ({grid_size}x{grid_size} pairs)...")
    
    # Compute loop with tqdm progress bar
    for i, h1 in enumerate(tqdm(h_vals, desc="Evaluating h1")):
        for j, h2 in enumerate(h_vals):
            is_straightforward = check_nolips_straightforward(h1, h2, L=1.0, D=1.0, delta=delta)
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
    plt.title(rf"NoLips Step Patterns (Certified Dots Only, $\Delta = {delta}$)", fontsize=11)
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
if __name__ == "__main__":
    run_nolips_grid_search()