import os

base_path = r"C:\Users\brook\Downloads\PythonProject\PythonProject\loss_history_GA\GA_Settings"
file1 = os.path.join(base_path, "optimization_summary.csv")
file2 = os.path.join(base_path, "part1_optimization_summary.csv")

def read_lines(filepath):
    with open(filepath, 'r') as f:
        return f.readlines()

lines1 = read_lines(file1)
lines2 = read_lines(file2)

# lines1 has header and content
# lines2 has header and content
# We want header + content1 + content2

header = lines1[0]
content1 = lines1[1:]
content2 = lines2[1:]

# Make sure the last line of content1 has a newline if it doesn't
if content1 and not content1[-1].endswith('\n'):
    content1[-1] += '\n'

combined_lines = [header] + content1 + content2

with open(file1, 'w') as f:
    f.writelines(combined_lines)

print(f"Merged {len(content2)} lines from part1 into optimization_summary.csv. Total lines: {len(combined_lines)}")

