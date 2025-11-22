import numpy as np
from scipy.optimize import differential_evolution, minimize

# Given constants
# alpha = np.array([0.33, 0.33, 0.33])
# beta = np.array([0.3, 0.3, 0.3])
# r = np.array([0.7, 0.7, 0.7])
# v0 = 1
# M = 10_000
# omega = 0.01  # weight for variance penalty
# p_t = 14

# p_min, p_max = 1, 100


from parameters import Parameters
param = Parameters.from_toml("scenario1.toml", "scenario1")

print("Scenario 1")
print(param)
print("")

_ = param.optimize_mnl()

param = Parameters.from_toml("scenario2.toml", "scenario2")
print("Scenario 2")
print(param)
print("")

_ = param.optimize_mnl()

param = Parameters.from_toml("scenario3.toml", "scenario3")
print("Scenario 3")
print(param)
print("")

_ = param.optimize_mnl()

param = Parameters.from_toml("scenario4.toml", "scenario4")
print("Scenario 4")
print(param)
print("")

_ = param.optimize_mnl()

param = Parameters.from_toml("scenario5.toml", "scenario5")
print("Scenario 5")
print(param)
print("")

_ = param.optimize_mnl()


param = Parameters.from_toml("scenario6.toml", "scenario6")
print("Scenario 6")
print(param)
print("")

_ = param.optimize_mnl()
