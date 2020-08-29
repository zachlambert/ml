import sys
sys.path.append("/home/zach/code/projects/machine_learning/ml_library/")
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

reg.plot_data_one_feature(None, r, 0)
plt.show()
reg.plot_data_two_features(None, r, 0, 1)
plt.show()

r.add_model(reg.ModelLinear())

r.fit()

print("===== RESULTS =====")
print(reg.describe_results(r))

reg.plot_result_one_feature(None, r, 0)
plt.show()
reg.plot_result_one_feature(None, r, 1)
plt.show()
