from dataclasses import dataclass

import numpy as np
import toml
import tomllib
from scipy.optimize import differential_evolution


@dataclass
class Parameters:
    alpha: np.ndarray
    beta1: np.ndarray
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
    def mnl(pi, alpha_i, beta_i):
        """Attractiveness from MNL model."""
        return np.exp(alpha_i - (beta_i * pi))

    def demand_mnl(self, p):
        """Expected demand for each platform (MNL)."""
        v = self.mnl(p, self.alpha, self.beta1)
        total_v = self.v0 + np.sum(v)
        return self.M * (v / total_v)

    # ---------- Revenue Functions ----------
    def revenue_mnl(self, p):
        """Expected revenue for each platform (MNL)."""
        d = self.demand_mnl(p)
        return d * p * self.r

    # ---------- Objective Functions ----------
    def objective_mnl(self, p):
        """Objective function to maximize (minimize its negative)."""
        R = self.revenue_mnl(p)
        var_penalty = np.var(R)
        total_R = np.sum(R) - (self.omega * var_penalty)
        return -total_R

    # ---------- Optimization ----------
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
