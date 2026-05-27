import os
import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns

def analyze():
    csv_path = os.path.join("loss_history_ES", "ES_Settings", "region_aware_es_optimization_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Identifiers for a configuration
    config_cols = ["sigma_high", "offspring_lambda", "alpha", "population_size_P", "ratio_and_mu"]
    df["Config_ID"] = df[config_cols].apply(lambda x: "_".join(x.astype(str)), axis=1)

    # The columns that represent different test settings (models + functions)
    # Exclude the configuration columns and the summary columns (avg_loss, avg_regions_discovered, avg_runtime_seconds)
    summary_cols = ["avg_loss", "avg_regions_discovered", "avg_runtime_seconds"]
    test_setting_cols = [c for c in df.columns if c not in config_cols and c not in summary_cols and c != "Config_ID"]

    # 1. Rank configurations per test setting
    # For each setting, lower loss is better, so rank them in ascending order
    ranks_df = pd.DataFrame()
    ranks_df["Config_ID"] = df["Config_ID"]

    for col in test_setting_cols:
        ranks_df[col] = df[col].rank(method="average", ascending=True)

    # Calculate average rank and rank variance (consistency) across test settings
    df["Avg_Rank"] = ranks_df[test_setting_cols].mean(axis=1)
    df["Rank_Variance"] = ranks_df[test_setting_cols].var(axis=1)
    df["Rank_Std"] = ranks_df[test_setting_cols].std(axis=1)

    # Sort df by average rank (primary) and then average loss (secondary)
    df_sorted = df.sort_values(by=["Avg_Rank", "avg_loss"], ascending=[True, True])

    print("=== Top 10 Configurations by Average Rank ===")
    top10_cols = config_cols + ["Avg_Rank", "Rank_Std", "avg_loss", "avg_regions_discovered"]
    print(df_sorted[top10_cols].head(10).to_string(index=False))

    # 2. Friedman Test
    # Data shape for Friedman: N datasets (rows), k algorithms (cols)
    # Our ranks_df currently has k algorithms (rows) and N datasets (cols)
    # We transpose it to match the expected format for the rank test
    data_for_test = df.set_index("Config_ID")[test_setting_cols].T
    
    # Check if there are at least 3 datasets and 3 configurations, otherwise can't do Friedman effectively
    if data_for_test.shape[1] >= 3:
        stat, p_value = stats.friedmanchisquare(*[data_for_test[col] for col in data_for_test.columns])
        print(f"\nFriedman Test Statistic: {stat:.4f}, p-value: {p_value:.4e}")

        if p_value < 0.05:
            print("Significant differences found globally across configurations.")
            # Nemenyi Test
            nem_results = sp.posthoc_nemenyi_friedman(data_for_test.values)
            nem_results.columns = data_for_test.columns
            nem_results.index = data_for_test.columns
            
            # Print significant differences for the top 1
            top1_config = df_sorted.iloc[0]["Config_ID"]
            print(f"\nComparisons against the best configuration ({top1_config}):")
            top1_p_values = nem_results[top1_config].sort_values()
            
            sig_diff = top1_p_values[top1_p_values < 0.05]
            non_sig_diff = top1_p_values[top1_p_values >= 0.05]
            
            print(f"- Significantly different (p < 0.05): {len(sig_diff)} configurations")
            print(f"- Not significantly different (p >= 0.05): {len(non_sig_diff) - 1} configurations") # -1 to exclude self

    # Identify conditionally best settings (best per architecture/dataset)
    print("\n=== Best Configuration Per Architecture/Dataset ===")
    for col in test_setting_cols:
        best_idx = ranks_df[col].idxmin()
        best_config = ranks_df.loc[best_idx, "Config_ID"]
        best_loss = df.loc[best_idx, col]
        print(f"Setting: {col}")
        print(f"  Best Config: {best_config} (Loss: {best_loss:.4f})")
    
    print("\n=== Best Configurations for 'disc' architecture specifically ===")
    disc_cols = [c for c in test_setting_cols if "disc_" in c]
    if disc_cols:
        df["Avg_Rank_Disc"] = ranks_df[disc_cols].mean(axis=1)
        disc_sorted = df.sort_values(by=["Avg_Rank_Disc", "avg_loss"], ascending=[True, True])
        top5_disc = config_cols + ["Avg_Rank_Disc", "avg_loss", "avg_regions_discovered"]
        print(disc_sorted[top5_disc].head(5).to_string(index=False))

    # 3. Create output directory for plots
    os.makedirs("analysis_plots", exist_ok=True)
    
    # Plot 1: Rank Distribution
    plt.figure(figsize=(12, 6))
    rank_melted = ranks_df.melt(id_vars=["Config_ID"], value_vars=test_setting_cols, 
                                var_name="Test_Setting", value_name="Rank")
    
    # Sort configs by average rank for the plot
    config_order = df_sorted["Config_ID"].tolist()
    
    sns.boxplot(data=rank_melted, x="Config_ID", y="Rank", order=config_order[:15])
    plt.xticks(rotation=90)
    plt.title("Rank Distribution of Top 15 Configurations across datasets")
    plt.tight_layout()
    plt.savefig("analysis_plots/rank_distribution.png")
    plt.close()
    
    # Plot 2: Regions Discovered
    plt.figure(figsize=(12, 6))
    top15_df = df_sorted.head(15)
    sns.barplot(data=top15_df, x="Config_ID", y="avg_regions_discovered", order=config_order[:15])
    plt.xticks(rotation=90)
    plt.title("Average Regions Discovered for Top 15 Configurations")
    plt.tight_layout()
    plt.savefig("analysis_plots/regions_discovered.png")
    plt.close()

    print("\nAnalysis complete. Plots saved to analysis_plots/")
    
if __name__ == "__main__":
    analyze()
#While I have no clue about what to write here, I will procrastinate and write some random text to fill the space. This is just a placeholder and has no meaning. I hope this helps me avoid writing actual code for a little while longer. Maybe I can even write some more random text to make it look like I'm doing something productive. But in reality, I'm just avoiding the task at hand. Oh well, I guess I'll have to get to it eventually.
#There is so much work
