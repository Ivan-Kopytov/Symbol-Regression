import os
import re
import argparse
import numpy as np
import sympy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

# определяем стиль
mpl.rcParams.update({
    "figure.figsize": (8.5, 6.0),
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
})

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

def parse_solution_best(solution_path: str):
    best = None
    with open(solution_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            last_num_idx = None
            for i, tok in enumerate(parts):
                if re.fullmatch(FLOAT, tok):
                    last_num_idx = i
            if last_num_idx is None or last_num_idx == len(parts) - 1:
                continue
            expr = " ".join(parts[last_num_idx + 1:])
            try:
                err = float(parts[last_num_idx])
                comp = float(parts[last_num_idx - 1]) if last_num_idx - 1 >= 0 else np.nan
            except Exception:
                continue
            if best is None or (np.isfinite(err) and err < best[0]):
                best = (err, comp, expr)
    if best is None:
        raise RuntimeError(f"No parsable lines in {solution_path}")
    return best

def load_table(path: str):
    data = np.loadtxt(path, delimiter=None, ndmin=2)
    if data.shape[1] < 2:
        raise ValueError("Need at least 2 columns: y and one feature.")
    y = data[:, 0]
    X = data[:, 1:]
    return y, X

def build_symbol_map(expr: sp.Expr, X: np.ndarray):
    free = sorted(expr.free_symbols, key=lambda s: str(s))
    k = X.shape[1]
    candidates = [
        [sp.Symbol(f"x{i}") for i in range(1, k+1)],
        [sp.Symbol(f"x{i}") for i in range(0, k)],
        [sp.Symbol(chr(ord('a')+i)) for i in range(k)],
        [sp.Symbol(f"v{i}") for i in range(1, k+1)],
    ]
    names = None
    for cand in candidates:
        if set(free).issubset(set(cand)):
            names = cand; break
    if names is None:
        names = free
    ordered = sorted(list(set(names) & set(free)), key=lambda s: str(s))
    if len(ordered) != len(free):
        ordered = free
    if len(ordered) > k:
        raise ValueError(f"Expression uses {len(ordered)} variables but X has {k} columns.")
    mapping = {s: X[:, i] for i, s in enumerate(ordered)}
    return ordered, mapping

def finite_mask(*arrs):
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m

def predict(expr, mapping, syms):
    f = sp.lambdify(tuple(syms), expr, modules=["numpy"])
    args = [mapping[s] for s in syms]
    return np.array(f(*args), dtype=float).reshape(-1)

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Plot points and model functional dependence (AI-Feynman).")
    ap.add_argument("--data_dir", default="./example_data")
    ap.add_argument("--filename", default="example1.txt")
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--out_dir", default="./results/plots_function")
    ap.add_argument("--grid", type=int, default=400, help="Grid points for smooth curve/surface")
    ap.add_argument("--heatmap_bins", type=int, default=200, help="Heatmap resolution for 2D")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    data_path = os.path.join(args.data_dir, args.filename)
    solution_path = os.path.join(args.results_dir, f"solution_{args.filename}")
    base = os.path.splitext(args.filename)[0]

    # 1) load and model
    err, comp, expr_str = parse_solution_best(solution_path)
    expr = sp.sympify(expr_str, convert_xor=True)
    y, X = load_table(data_path)

    syms, mapping = build_symbol_map(expr, X)
    y_pred = predict(expr, mapping, syms)
    mask = finite_mask(y, y_pred)
    y, X, y_pred = y[mask], X[mask], y_pred[mask]

    k = X.shape[1]
    if k == 1:
        x = X[:, 0]
        xg = np.linspace(np.min(x), np.max(x), args.grid)
        map_grid = {syms[0]: xg}
        yg = predict(expr, map_grid, [syms[0]])
        order = np.argsort(xg)
        xg, yg = xg[order], yg[order]

        plt.figure(figsize=(8.5, 6))
        plt.scatter(x, y, s=14, alpha=0.7, label="data")
        plt.plot(xg, yg, linewidth=2.5, label="model")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{base}: data and model curve (k=1)")
        plt.legend()
        path = os.path.join(args.out_dir, f"{base}_curve_1d.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print("saved:", path)

    elif k == 2:
        x1, x2 = X[:, 0], X[:, 1]
        g1 = np.linspace(np.min(x1), np.max(x1), int(np.sqrt(args.grid)))
        g2 = np.linspace(np.min(x2), np.max(x2), int(np.sqrt(args.grid)))
        G1, G2 = np.meshgrid(g1, g2, indexing="xy")

        map_grid = {syms[0]: G1, syms[1]: G2}
        Z = predict(expr, map_grid, syms[:2])

        # 3D surface
        fig = plt.figure(figsize=(9, 6.8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(G1, G2, Z, rstride=1, cstride=1, linewidth=0, alpha=0.6)
        ax.scatter(x1, x2, y, s=10, alpha=0.7)
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("y")
        ax.set_title(f"{base}: data and model surface (k=2)")
        path3d = os.path.join(args.out_dir, f"{base}_surface_3d.png")
        plt.savefig(path3d, bbox_inches="tight"); plt.close(fig)
        print("saved:", path3d)

        # heatmap (2D)
        plt.figure(figsize=(8.5, 6.5))
        extent = [g1.min(), g1.max(), g2.min(), g2.max()]
        plt.imshow(Z.T, origin="lower", extent=extent, aspect="auto")
        plt.scatter(x1, x2, c=y, s=12, edgecolor="k", linewidth=0.2, alpha=0.8)
        plt.xlabel("x1"); plt.ylabel("x2"); plt.title(f"{base}: model heatmap + points")
        pathhm = os.path.join(args.out_dir, f"{base}_heatmap_2d.png")
        plt.savefig(pathhm, bbox_inches="tight"); plt.close()
        print("saved:", pathhm)

    else:
        med = np.nanmedian(X, axis=0)
        for j in range(k):
            
            xj = X[:, j]
            xg = np.linspace(np.min(xj), np.max(xj), args.grid)
            grid_map = {}
            for i, s in enumerate(syms):
                if i == j:
                    grid_map[s] = xg
                else:
                    grid_map[s] = np.full_like(xg, med[i], dtype=float)
            yg = predict(expr, grid_map, syms)
            order = np.argsort(xg)
            xg, yg = xg[order], yg[order]

            plt.figure(figsize=(8.5, 6))
            plt.scatter(xj, y, s=12, alpha=0.6, label="data")
            plt.plot(xg, yg, linewidth=2.5, label="model (others=median)")
            plt.xlabel(f"x{j}"); plt.ylabel("y")
            plt.title(f"{base}: partial dependence for x{j} (k={k})")
            plt.legend()
            path = os.path.join(args.out_dir, f"{base}_partial_x{j}.png")
            plt.savefig(path, bbox_inches="tight"); plt.close()
            print("saved:", path)

    resid = y - y_pred
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    print(f"MAE={mae:.4g}, RMSE={rmse:.4g}, n={len(y)}")

if __name__ == "__main__":
    import numpy as _np
    assert int(_np.__version__.split('.')[0]) == 1, f"Detected NumPy={_np.__version__}. Please install numpy<2."
    main()