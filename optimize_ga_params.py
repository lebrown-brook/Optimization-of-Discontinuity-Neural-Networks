import itertools
import os
import pandas as pd

from experiment_runner_GA import run_ga_experiments

def optimize_ga():
    # Grid definition

    num_parents_mating_options = [10, 15]
    keep_elitism_options = [1, 2, 3, 4]
    # Selection: 'sss' or 'tournament'. If tournament, vary K.
    selection_options = [
        {"parent_selection_type": "sss"},
        {"parent_selection_type": "tournament", "K_tournament": 40}
    ]
    crossover_type_options = ["scattered", "single_point"]
    mutation_rate_options = [[10, 2], [5, 1]] # Adaptive mutation rates: [P_high, P_low]
    mutation_rate_options = [[a[0]/100, a[1]/100] for a in mutation_rate_options]  # Convert to percentages

    # Iterate
    configs_list = []

    products = itertools.product(
        num_parents_mating_options,
        keep_elitism_options,
        crossover_type_options,
        mutation_rate_options
    )

    for npm, ke, ct, mr in products:
        base_conf = {
            "num_parents_mating": npm,
            "keep_elitism": ke,
            "crossover_type": ct,
            "mutation_probability": mr
        }

        for sel in selection_options:
            conf = base_conf.copy()
            conf.update(sel)
            configs_list.append(conf)

    print(f"Total configurations to run: {len(configs_list)}")

    all_summary_data = []

    for i, conf in enumerate(configs_list):
        # Create a descriptive string for filename
        sel_str = conf["parent_selection_type"]
        if sel_str == "tournament":
            sel_str += f"_k{conf['K_tournament']}"

        mr_str = str(conf['mutation_probability']).replace('[', '').replace(']', '').replace(', ', '_')
        conf_str = f"npm_{conf['num_parents_mating']}_ke_{conf['keep_elitism']}_sel_{sel_str}_ct_{conf['crossover_type']}_mr_{mr_str}"

        print(f"[{i+1}/{len(configs_list)}] Running config: {conf_str}")

        save_dir = os.path.join("loss_history_GA", "GA_Settings")
        os.makedirs(save_dir, exist_ok=True)

        batch_results = run_ga_experiments(
            ga_settings_override=conf,
            save_dir=save_dir,
            enable_plotting=False,
            filename_prefix=f"pwlu_adaptive_bigPopulation_{conf_str}_"
        )

        # Collect data for summary
        for res in batch_results:
             row = conf.copy()
             # Ensure K_tournament exists if not present (for non-tournament to avoid creating multiple separate columns or errors)
             if "K_tournament" not in row:
                 row["K_tournament"] = "N/A"

             # Create a unique model identifier
             # Combining variation, function name and architecture
             model_id = f"{res['variation']}_{res['function']}_{res['architecture']}"

             row['model_id'] = model_id
             row['final_loss'] = res['final_loss']
             all_summary_data.append(row)

    # Generate and save summary table
    if all_summary_data:
        df = pd.DataFrame(all_summary_data)

        # Convert mutation_probability list to string so it can be used in pivot_table index
        df['mutation_rate_str'] = df['mutation_probability'].apply(str)

        # Define index columns (GA params)
        index_cols = ["num_parents_mating", "keep_elitism", "parent_selection_type", "K_tournament", "crossover_type", "mutation_rate_str"]

        # Create pivot table: Index=GA Params, Columns=Model, Values=Loss
        # aggfunc='first' should be fine as we have 1 run per config per model
        pivot_df = df.pivot_table(index=index_cols, columns='model_id', values='final_loss', aggfunc='first')

        # Add average loss column for sorting
        pivot_df['avg_loss'] = pivot_df.mean(axis=1)

        #add column that calculates absolute of difference between the loss columns
        if pivot_df.shape[1] >= 2:
            col1 = pivot_df.columns[0]
            col2 = pivot_df.columns[1]
            pivot_df['loss_diff'] = (pivot_df[col1] - pivot_df[col2]).abs()

        # Sort by average loss (ascending, since lower loss is better)
        pivot_df = pivot_df.sort_values('avg_loss')

        print("\n=== Optimization Summary Table (Sorted by Avg Final Loss) ===")
        print(pivot_df)

        # Save to CSV
        summary_dir = os.path.join("loss_history_GA", "GA_Settings")
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, "pwlu_scattered_vs_singlepoint_adaptive_optimization_summary.csv")
        pivot_df.to_csv(summary_path)
        print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    optimize_ga()

