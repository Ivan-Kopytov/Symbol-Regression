import os, re, csv, time, pathlib
import numpy as np
import aifeynman as af

DATA_DIR = "./example_data"  
RESULTS_DIR = "./results"
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")

BF_TRY_TIME = 60              
BF_OPS = "14ops.txt"
POLY_DEG = 3
NN_EPOCHS = 500              

#стркои решения
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

def parse_solution_line(line: str):
    parts = line.strip().split()
    if not parts:
        return None
    # индекс последнего числа
    last_num_idx = None
    for i, tok in enumerate(parts):
        if re.fullmatch(FLOAT, tok):
            last_num_idx = i
    if last_num_idx is None or last_num_idx == len(parts) - 1:
        return None
    expr = " ".join(parts[last_num_idx+1:])
    try:
        err = float(parts[last_num_idx])
        complexity = float(parts[last_num_idx-1]) if last_num_idx - 1 >= 0 else np.nan
    except Exception:
        err, complexity = np.nan, np.nan
    return err, complexity, expr

def pick_best_from_solution_file(path: str):
    best = None  
    with open(path, "r") as f:
        for line in f:
            parsed = parse_solution_line(line)
            if not parsed:
                continue
            err, comp, expr = parsed
            if best is None or (np.isfinite(err) and err < best[0]):
                best = (err, comp, expr)
    return best

def main():
    import numpy as _np
    assert int(_np.__version__.split('.')[0]) == 1, f"Detected NumPy={_np.__version__}. Please install numpy<2."

    pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    if os.path.basename(DATA_DIR) == "example_data":
        af.get_demos(DATA_DIR)

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".txt") or f.endswith(".csv")])
    print(f"Найдено датасетов: {len(files)}")

    rows = []
    for fname in files:
        print(f"\n=== {fname} ===")
        t0 = time.time()
        try:
            af.run_aifeynman(
                pathdir=DATA_DIR if DATA_DIR.endswith("/") else DATA_DIR + "/",
                filename=fname,
                BF_try_time=BF_TRY_TIME,
                BF_ops_file_type=BF_OPS,
                polyfit_deg=POLY_DEG,
                NN_epochs=NN_EPOCHS
            )
        except Exception as e:
            print(f"[FAIL] {fname}: {e}")
            rows.append({"dataset": fname, "best_error": "FAIL", "complexity_bits": "", "expr": "", "solution_file": ""})
            continue

        sol_path = os.path.join(RESULTS_DIR, f"solution_{fname}")
        if not os.path.exists(sol_path):
            print(f"[WARN] solution не найден для {fname}")
            rows.append({"dataset": fname, "best_error": "NO_SOLUTION", "complexity_bits": "", "expr": "", "solution_file": ""})
            continue

        best = pick_best_from_solution_file(sol_path)
        if best is None:
            print(f"[WARN] пустой или непарсабельный solution: {sol_path}")
            rows.append({"dataset": fname, "best_error": "EMPTY", "complexity_bits": "", "expr": "", "solution_file": sol_path})
        else:
            err, comp, expr = best
            dt = time.time() - t0
            print(f"Best: err={err:.6g}, complexity={comp:.3g}, time={dt:.1f}s")
            rows.append({"dataset": fname, "best_error": err, "complexity_bits": comp, "expr": expr, "solution_file": sol_path})

    # пишем сводку
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset","best_error","complexity_bits","expr","solution_file"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n Сводка: {SUMMARY_CSV}")

if __name__ == "__main__":
    main()