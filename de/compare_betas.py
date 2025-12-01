import os

import matplotlib.pyplot as plt
import numpy as np

from parameters import Parameters


def run_comparison():
    # Define the range of base beta values to test
    # We test from 0.1 to 1.0.
    # This value 'b' acts as the baseline beta.
    # For uniform scenarios (1-5), all platforms get beta = b.
    # For Scenario 6, the platform with the lowest sensitivity gets b,
    # and others are shifted up to maintain the original spread/pattern.
    betas_to_test = np.linspace(0.1, 1.0, 10)

    scenario_files = [
        "scenario1.toml",
        "scenario2.toml",
        "scenario3.toml",
        "scenario4.toml",
        "scenario5.toml",
        "scenario6.toml"
    ]

    # Dictionary to store results: {scenario_name: {platform_idx: [prices]}}
    results = {}

    print("Starting beta comparison...")

    for s_file in scenario_files:
        scenario_name = s_file.replace(".toml", "")
        print(f"\n--- Processing {scenario_name} ---")

        # Load parameters to determine the beta structure
        base_param = Parameters.from_toml(s_file, scenario_name)
        num_platforms = len(base_param.alpha)

        # Initialize storage for this scenario
        # We want a list of prices for each platform across the beta range
        scenario_prices = {i: [] for i in range(num_platforms)}

        # Calculate offsets relative to the minimum beta in the original file.
        original_betas = base_param.beta1
        min_beta = np.min(original_betas)
        offsets = original_betas - min_beta

        for b in betas_to_test:
            # Create a fresh parameter object
            param = Parameters.from_toml(s_file, scenario_name)

            # Apply the new beta values
            # We shift the entire beta vector so that its minimum aligns with 'b'.
            param.beta1 = b + offsets

            try:
                # Run optimization
                # Using lower popsize/maxiter for speed in this loop
                res = param.optimize_mnl(verbose=False, popsize=15, maxiter=200)

                # Store the optimal price for each platform
                for i in range(num_platforms):
                    scenario_prices[i].append(res.x[i])

            except Exception as e:
                print(f"Optimization failed for beta={b}: {e}")
                for i in range(num_platforms):
                    scenario_prices[i].append(0)

        results[scenario_name] = scenario_prices

    # Generate Plot: Grid of subplots (2 columns, 3 rows)
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.flatten()

    for idx, s_file in enumerate(scenario_files):
        scenario_name = s_file.replace(".toml", "")
        ax = axes[idx]

        scenario_data = results[scenario_name]

        for platform_idx, prices in scenario_data.items():
            ax.plot(betas_to_test, prices, marker='.', label=f"Platform {platform_idx+1}")

        ax.set_title(f"{scenario_name}")
        ax.set_xlabel("Base Beta (Sensitivity)")
        ax.set_ylabel("Optimal Price")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(fontsize='small')

    plt.tight_layout()

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    output_path = "results/beta_comparison_prices_per_platform.png"
    plt.savefig(output_path)
    print(f"\nComparison complete. Graph saved to {output_path}")

if __name__ == "__main__":
    run_comparison()
