import pandas as pd

#add a column to an existing csv file adaptive_scattered_bigPopulation_optimization_summary.csv which calculates absolute of difference between two loss columns
file_path = r"C:\Users\brook\Downloads\PythonProject\PythonProject\loss_history_GA\GA_Settings\adaptive_scattered_bigPopulation_optimization_summary.csv"
df = pd.read_csv(file_path)
#loss columns are named cont_linear_with_varying_slopes_[1, 16, 8, 4] and disc_linear_with_varying_slopes_[1, 16, 8, 4]
col1 = 'cont_linear_with_varying_slopes_[1, 16, 8, 4]'
col2 = 'disc_linear_with_varying_slopes_[1, 16, 8, 4]'
df['loss_diff'] = (df[col1] - df[col2]).abs()
df.to_csv(file_path, index=False)
print(f"Added loss_diff column to {file_path}")
