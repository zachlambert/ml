import sys
sys.path.append("../") # If run from within test directory
from ml import regression as reg
import matplotlib.pyplot as plt

D = 2

d = reg.ArtificialDataset()
d.generate_X_gaussian(D, 1000)
weights, noise_var = d.generate_y_linear()

r = reg.Manager(D)
r.load_data(d.X, d.y)

print("===== DATA =====")
print(reg.describe_data(r))

fig, [ax1, ax2] = plt.subplots(1, 2)
reg.plot_data_one_feature(ax1, r, 0)
reg.plot_data_two_features(ax2, r, 0, 1)
plt.show()

r.add_model(reg.ModelLinear())

r.fit()

print("===== RESULTS =====")
print(reg.describe_results(r))

fig, [ax1, ax2] = plt.subplots(1, 2)
reg.plot_results_one_feature(ax1, r, 0)
reg.plot_results_one_feature(ax2, r, 1)
plt.show()
