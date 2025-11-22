import numpy as np
import tomllib

from parameters import Parameters


def show_platform_revenues():
    """Show individual platform revenues from saved optimization results."""

    scenarios = ["scenario1", "scenario2", "scenario3", "scenario4", "scenario5", "scenario6"]

    print("PLATFORM REVENUE BREAKDOWN")
    print("=" * 80)

    for scenario in scenarios:
        print(f"\n{scenario.upper()}")
        print("-" * 50)

        # Load parameters
        param = Parameters.from_toml(f"{scenario}.toml", scenario)

        for model in ["mci", "mnl"]:
            try:
                # Load results
                with open(f"results/{scenario}_{model}.toml", "rb") as f:
                    results = tomllib.load(f)

                # Parse prices from string format "(a, b, c, ...)"
                price_str = results["Price"].strip("()")
                prices = np.array([float(x.strip()) for x in price_str.split(",")])

                # Calculate platform revenues
                demands = param.demand_mnl(prices)
                platform_revenues = demands * prices * param.r

                print(f"\n{model.upper()} Model:")
                print(f"Total Revenue: {np.sum(platform_revenues):.2f}")
                print("Individual Platform Revenues:")

                for i, (price, demand, revenue) in enumerate(zip(prices, demands, platform_revenues)):
                    print(f"  Platform {i+1}: ${revenue:>8.2f} (Price: ${price:>6.2f}, Demand: {demand:>7.1f})")

            except Exception as e:
                print(f"Error processing {scenario}_{model}: {e}")

if __name__ == "__main__":
    show_platform_revenues()
