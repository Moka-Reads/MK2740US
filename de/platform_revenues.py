import numpy as np
import tomllib

from parameters import Parameters


def calculate_platform_revenues():
    """Calculate individual platform revenues for all scenarios and models."""

    scenarios = ["scenario1", "scenario2", "scenario3", "scenario4", "scenario5", "scenario6"]

    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"SCENARIO: {scenario.upper()}")
        print(f"{'='*50}")

        # Load parameters
        param = Parameters.from_toml(f"{scenario}.toml", scenario)

        # Load optimization results
        for model in ["mci", "mnl"]:
            try:
                with open(f"results/{scenario}_{model}.toml", "rb") as f:
                    results = tomllib.load(f)

                # Parse prices from results
                price_str = results["Price"]
                # Remove parentheses and split by comma
                prices = [float(x.strip()) for x in price_str.strip("()").split(",")]
                prices = np.array(prices)

                print(f"\n{model.upper()} MODEL:")
                print(f"Optimal Prices: {prices}")

                # Calculate demands and revenues per platform
                demands = param.demand_mnl(prices)
                revenues = param.revenue_mnl(prices)

                print(f"Platform Demands: {demands}")
                print(f"Platform Revenues: {revenues}")
                print(f"Total Revenue: {np.sum(revenues):.4f}")
                print(f"Revenue Verification: {results['Revenue']:.4f}")

                # Print individual platform breakdown
                print("\nPlatform Breakdown:")
                for i, (p, d, r) in enumerate(zip(prices, demands, revenues)):
                    print(f"  Platform {i+1}: Price={p:.4f}, Demand={d:.2f}, Revenue={r:.2f}")

            except FileNotFoundError:
                print(f"Results file not found for {scenario}_{model}")
                continue

if __name__ == "__main__":
    calculate_platform_revenues()
