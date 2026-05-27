import os

base_path = r"C:\Users\brook\Downloads\PythonProject\PythonProject\loss_history_GA\GA_Settings"
file1 = os.path.join(base_path, "optimization_summary.csv")

try:
    with open(file1, 'r') as f:
        lines = f.readlines()
        print(f"Total lines: {len(lines)}")
        print("First 3 lines:")
        for l in lines[:3]:
            print(l.strip())
        print("Last 3 lines:")
        for l in lines[-3:]:
            print(l.strip())
except Exception as e:
    print(f"Error reading file: {e}")

