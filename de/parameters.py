from dataclasses import dataclass
import tomllib
import numpy as np
from scipy.optimize import differential_evolution
import toml
@dataclass
class Parameters:
    alpha: np.ndarray
    beta1: np.ndarray
    beta2: np.ndarray
    r: np.ndarray
    v0: float
    M: int
    omega: float
    p_min: float
    p_max: float
    name: str

    @classmethod
    def from_toml(cls, path: str, name: str):
        """Load parameters from TOML file and convert lists to NumPy arrays."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        params = data["parameters"]

        return cls(
            name=name,
            alpha=np.array(params["alpha"], dtype=float),
            beta1=np.array(params["beta1"], dtype=float),
            beta2=np.array(params["beta2"], dtype=float),
            r=np.array(params["r"], dtype=float),
            v0=float(params["v0"]),
            M=int(params["M"]),
            omega=float(params["omega"]),
            p_min=float(params["p_min"]),
            p_max=float(params["p_max"]),
        )
    def __str__(self):
        lines = ["\nParameters:"]
        for k, v in vars(self).items():
            lines.append(f"{k:<10} : {v}")
        return "\n".join(lines)


    # ---------- Model Functions ----------
    @staticmethod
    def mci(pi, alpha_i, beta_i):
        """Attractiveness from MCI model."""
        return alpha_i * (pi ** -beta_i)

    @staticmethod
    def mnl(pi, alpha_i, beta_i):
        """Attractiveness from MNL model."""
        return np.exp(alpha_i - (beta_i * pi))

    # ---------- Demand Functions ----------
    def demand_mci(self, p):
        """Expected demand for each platform (MCI)."""
        v = self.mci(p, self.alpha, self.beta2)
        total_v = self.v0 + np.sum(v)
        return self.M * (v / total_v)

    def demand_mnl(self, p):
        """Expected demand for each platform (MNL)."""
        v = self.mnl(p, self.alpha, self.beta1)
        total_v = self.v0 + np.sum(v)
        return self.M * (v / total_v)

    # ---------- Revenue Functions ----------
    def revenue_mci(self, p):
        """Expected revenue for each platform (MCI)."""
        d = self.demand_mci(p)
        return d * p * self.r

    def revenue_mnl(self, p):
        """Expected revenue for each platform (MNL)."""
        d = self.demand_mnl(p)
        return d * p * self.r

    # ---------- Objective Functions ----------
    def objective_mci(self, p):
        """Objective function to maximize (minimize its negative)."""
        R = self.revenue_mci(p)
        var_penalty = np.var(R)
        total_R = np.sum(R) - (self.omega * var_penalty)
        return -total_R

    def objective_mnl(self, p):
        """Objective function to maximize (minimize its negative)."""
        R = self.revenue_mnl(p)
        var_penalty = np.var(R)
        total_R = np.sum(R) - (self.omega * var_penalty)
        return -total_R

    # ---------- Optimization ----------
    def optimize_mci(self):
        """Run differential evolution for MCI model."""
        bounds = [(self.p_min, self.p_max) for _ in range(len(self.alpha))]
        result = differential_evolution(
            self.objective_mci, bounds, popsize=50, maxiter=1000, tol=1e-6
        )
        print("(MCI) Optimal Prices:", result.x)
        print("(MCI) Optimal Revenue:", -result.fun)
        print("(MCI) Royalty/Platform:", self.r * result.x)

        # save as csv
        price_str = "(" + ", ".join(str(float(round(x, 4))) for x in result.x) + ")"
        royalty_str = "(" + ", ".join(str(float(round(x, 4))) for x in (self.r * result.x)) + ")"

        data = {
            "Price": price_str,
            "Revenue": float(round(-result.fun, 4)),
            "Royalty_Per_Platform": royalty_str,
        }


        with open(f"results/{self.name}_mci.toml", "w") as f:
            toml.dump(data, f)
        return result

    def optimize_mnl(self):
        """Run differential evolution for MNL model."""
        bounds = [(self.p_min, self.p_max) for _ in range(len(self.alpha))]
        result = differential_evolution(
            self.objective_mnl, bounds, popsize=50, maxiter=1000, tol=1e-6
        )
        print("(MNL) Optimal Prices:", result.x)
        print("(MNL) Optimal Revenue:", -result.fun)
        print("(MNL) Royalty/Platform:", self.r * result.x)
        price_str = "(" + ", ".join(str(float(round(x, 4))) for x in result.x) + ")"
        royalty_str = "(" + ", ".join(str(float(round(x, 4))) for x in (self.r * result.x)) + ")"

        data = {
            "Price": price_str,
            "Revenue": float(round(-result.fun, 4)),
            "Royalty_Per_Platform": royalty_str,
        }


        with open(f"results/{self.name}_mnl.toml", "w") as f:
            toml.dump(data, f)
        return result
