__author__ = 'prabhath'

import numpy as np
import matplotlib.pyplot as plt

""" Learning Principal component analysis. """

# 1. Sample data

x = np.array([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1])
y = np.array([2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9])

# 2. Adjust the data

x_ = [j - np.mean(x) for j in x]
y_ = [j - np.mean(y) for j in y]

''' sample plot of data '''

plt.plot(x, y, 'o', label="Original data")
plt.plot(x_, y_, 'x', label="Adjusted Data")
plt.axis([-3, 6, -3, 6])
plt.title("Original Data, Adjusted data and Transformed Plot")
plt.xlabel("Column-1 or $x_i$")
plt.ylabel("Column-2 or $y_i$")


# 3. Calculating covariance of the data by making a matrix of the adjusted data.

X = np.vstack((x_, y_))
cov = np.cov(X)

# 4. Calculate eigen values and eigen vectors of the matrix.

eig = np.linalg.eig(cov)
eig_values = eig[0]
eig_vector = eig[1]

eig_pairs = [(np.abs(eig_values[i]), eig_vector[:, i]) for i in range(len(eig_values))]
eig_pairs.sort()
eig_pairs.reverse()

# 5. Choosing components and forming feature vector.

row_feature_vector = np.hstack((eig_pairs[0][1].reshape(2, 1), eig_pairs[1][1].reshape(2, 1)))
row_data_adjust = X

final_data = (np.matrix(row_feature_vector).T * np.matrix(X)).T

f = [float(x) for x in final_data[:, 0]]
g = [float(x) for x in final_data[:, 1]]

plt.scatter(f, g, marker='*', label="Transformed Data")
plt.legend()
plt.show()
plt.savefig("Original Data, Adjusted data and Transformed Plot")

"""
Compare the variance of the two columns of the transformed_data choose the one with high variance.
In case of more than one column select the columns with variance above the threshold.
"""