# Symbol-Regression (AI-Feynman pipeline)

Проект для поиска аналитических формул по данным на основе идей AI-Feynman. Содержит полный пайплайн: генерация/загрузка демо-датасетов, запуск поиска, верификация решений и визуализация зависимостей.

---

## Возможности
- Запуск AI-Feynman с контролем времени brute-force и числа эпох NN.
- Автоматический парсинг solution_*.txt, выбор лучшей формулы по ошибке.
- Сводный отчёт results/summary.csv по всем датасетам.
- Визуализация: 1D кривая, 2D поверхность + heatmap, частные зависимости при k>2.
- Поддержка Git LFS для тяжёлых .h5 (по желанию).

---

## Структура репозитория
FEYNNAN/  
├─ example_data/                 # тестовые датасеты  
│  ├─ example1.txt  
│  ├─ example2.txt  
│  └─ example3.txt  
├─ run_feynman_all.py            # быстрый прогон одного датасета (demo)  
├─ verify.py                     # пакетный прогон, парсинг решений, summary.csv  
├─ viz_function_curve.py         # визуализация найденной функции  
├─ args.dat, constraints.txt     # конфигурация/ограничения для поиска  
├─ requirements.txt              # зависимости  
└─ results/                      # результаты (игнорируются git'ом)  
   ├─ solution_example1.txt  
   ├─ plots/  
   └─ plots_function/  

---

## Требования и установка
Рекомендуется Python ≥ 3.10. Виртуальное окружение:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Важно: NumPy должен быть 1.x (AI-Feynman не совместим с NumPy 2.x).  
Если установлен NumPy 2.x:

    pip install "numpy<2"

Опционально: Git LFS для моделей .h5:

    git lfs install
    git lfs track "*.h5"
    git add .gitattributes
    git commit -m "Track .h5 via LFS"

---

## Быстрый старт

### 1) Демонстрационный прогон одного датасета
Запусти:

    python run_feynman_all.py

Скрипт очистит старые артефакты, скачает/обновит example_data, запустит поиск по example1.txt и положит результат в:

    results/solution_example1.txt

---

### 2) Пакетная верификация по всем датасетам
Запусти:

    python verify.py

На выходе появится:

    results/summary.csv

Файл содержит столбцы: dataset, best_error, complexity_bits, expr, solution_file.

---

### 3) Визуализация найденной зависимости
Пример запуска:

    python viz_function_curve.py --filename example1.txt

Результаты:  
- для 1D: results/plots_function/example1_curve_1d.png  
- для 2D: results/plots_function/example1_surface_3d.png, example1_heatmap_2d.png  
- для k>2: results/plots_function/example1_partial_x*.png  

---

## Типовой рабочий сценарий
1. Подготовьте свои данные в формате таблицы: первый столбец — целевая y, далее признаки x1, x2, ...  
2. Поместите файл в example_data/ (или укажите свою папку флагом --data_dir).  
3. Запустите verify.py для пакетного прогона всех файлов и получения сводки.  
4. Визуализируйте интересующий датасет через viz_function_curve.py.  
5. Проанализируйте ошибку/сложность (best_error, complexity_bits) и итоговую формулу (expr).  

---

## Лицензия
MIT License © 2025 Ivan Kopytov
