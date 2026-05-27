import itertools
import os
import time

import pandas as pd

from experiments.experiment_runner_ES import run_es_experiments


def optimize_es():
    sigma_high_options = [0.1, 0.2, 0.4]
    sigma_low = 0.02
    
    # Pairs of (offspring_lambda, population_size_P) such that offspring * pop_size ~ 100
    lambda_pop_pairs = [(4, 25), (10, 10), (15, 7), (20, 5), (25, 4)]
    
    alpha_options = [0.01, 0.05]
    mu_ratios = [0.2, 0.5]

    configs_list = list(
        itertools.product(
            sigma_high_options,
            lambda_pop_pairs,
            alpha_options,
            mu_ratios,
        )
    )

    print(f"Total ES configurations to run: {len(configs_list)}")

    all_summary_data = []

    for i, (sigma_high, (offspring_lambda, pop_size), alpha, mu_ratio) in enumerate(configs_list):
        mu_per_region = max(1, int(pop_size * mu_ratio))
        ratio_and_mu = f"{mu_ratio}_{mu_per_region}"
        
        conf = {
            "sigma_low": sigma_low,
            "sigma_high": sigma_high,
            "offspring_lambda": offspring_lambda,
            "alpha": alpha,
            "population_size_P": pop_size,
            "mu_per_region": mu_per_region,
            "mu_ratio": mu_ratio,
            "ratio_and_mu": ratio_and_mu,
        }

        conf_str = (
            f"sigH_{sigma_high}_lambda_{offspring_lambda}_alpha_{alpha}_"
            f"pop_{pop_size}_muratio_{ratio_and_mu}"
        )

        print(f"[{i + 1}/{len(configs_list)}] Running config: {conf_str}")

        save_dir = os.path.join("loss_history_ES", "ES_Settings")
        os.makedirs(save_dir, exist_ok=True)

        start_time = time.time()
        batch_results = run_es_experiments(
            es_settings_override=conf,
            save_dir=save_dir,
            enable_plotting=False,
            filename_prefix=f"region_es_{conf_str}_",
        )
        runtime = time.time() - start_time

        for res in batch_results:
            row = conf.copy()
            model_id = f"{res['variation']}_{res['function']}_{res['architecture']}"
            row["model_id"] = model_id
            row["final_loss"] = res["final_loss"]
            row["regions_discovered"] = res.get("regions_discovered")
            row["runtime_seconds"] = runtime
            all_summary_data.append(row)

    if all_summary_data:
        df = pd.DataFrame(all_summary_data)

        index_cols = [
            "sigma_high",
            "offspring_lambda",
            "alpha",
            "population_size_P",
            "ratio_and_mu",
        ]

        loss_pivot = df.pivot_table(index=index_cols, columns="model_id", values="final_loss", aggfunc="first")
        loss_pivot["avg_loss"] = loss_pivot.mean(axis=1)
        loss_pivot = loss_pivot.sort_values("avg_loss")

        regions_mean = (
            df.groupby(index_cols, as_index=True)["regions_discovered"]
            .mean()
            .rename("avg_regions_discovered")
        )
        runtime_mean = (
            df.groupby(index_cols, as_index=True)["runtime_seconds"]
            .mean()
            .rename("avg_runtime_seconds")
        )
        loss_pivot = loss_pivot.join(regions_mean, how="left").join(runtime_mean, how="left")

        print("\n=== ES Optimization Summary Table (Sorted by Avg Final Loss) ===")
        print(loss_pivot)

        summary_dir = os.path.join("loss_history_ES", "ES_Settings")
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, "region_aware_es_optimization_summary.csv")
        loss_pivot.to_csv(summary_path)
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    optimize_es()
