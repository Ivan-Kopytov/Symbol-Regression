import os, sys, shutil, time, subprocess, inspect

def sh(cmd):
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=False, text=True, capture_output=False)

import numpy as np
import pandas, scipy, matplotlib, sklearn, sympy
import torch, torchvision
import aifeynman as af

print("NumPy:", np.__version__)
print("pandas:", pandas.__version__)
print("scipy:", scipy.__version__)
print("matplotlib:", matplotlib.__version__)
print("scikit-learn:", sklearn.__version__)
print("torch:", torch.__version__, "| MPS built:", torch.backends.mps.is_built(), "| MPS available:", torch.backends.mps.is_available())

# критично: NumPy должен быть 1.x
assert int(np.__version__.split('.')[0]) == 1, f"Нужен NumPy 1.x, сейчас {np.__version__}"

print("\n Очистка артефакты прошлых запусков")
for p in ("results_gen_sym.dat", "results"):
    if os.path.isdir(p):
        shutil.rmtree(p)
    elif os.path.isfile(p):
        os.remove(p)
print("OK")

print("\n Готовим данные для прогона")
af.get_demos("example_data")
print("OK: example_data готова")

print("\n Прогон данных (BF=8 c, NN=50 эпох)")
t0 = time.time()
af.run_aifeynman(
    pathdir="./example_data/",
    filename="example1.txt",
    BF_try_time=8,                 # укороченный brute-force для проверки
    BF_ops_file_type="14ops.txt",
    polyfit_deg=2,
    NN_epochs=50                   # укороченное обучение NN
)
dt = time.time() - t0
print(f"OK: demo завершено за {dt:.1f}s")

sol = "results/solution_example1.txt"
if os.path.exists(sol):
    print("Найден файл решения:", sol)
    with open(sol, "r") as f:
        print("Первые строки решения:")
        for i, line in enumerate(f):
            if not line.strip(): 
                continue
            print("  ", line.strip())
            if i >= 2:
                break
else:
    raise SystemExit(" solution_example1.txt не найден")

print("\n Проверка, что брутфорс был инициирован")
assert os.path.isdir("results"), "Папка results не создана"
print("Пайплан прошел")