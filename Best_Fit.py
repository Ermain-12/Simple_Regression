from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


# This function is used to generate random data rather than using the hard-coded data above
def create_data_set(hm, variance, step=2, correlation=False):
    val = 1
    ys = []

    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(x, y):
    m = (((mean(x)*mean(y)) - mean(x*y)) /
          ((mean(x) * mean(x)) - mean(x**2)))

    b = mean(y) - m*mean(x)

    return m, b


# Now, we calculate the coefficient of determination to determine how good of a fit is our best-fit line
def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regression = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)

    return 1 - (squared_error_regression / squared_error_y_mean)


# Create a data set
xs, ys = create_data_set(40, 5, 2, correlation='pos')

m, b = best_fit_slope_and_intercept(xs, ys)

# Create the linear regression line
regression_line = [(m * x) + b for x in xs]

# Coefficient of determination
r_squared = coefficient_of_determination(ys, regression_line)

# Print our coefficient of determination
print("Coefficient of Determination: ", r_squared)

# Make a sample prediction
x_predict = 8
y_predict = (m*x_predict) + b

plt.scatter(xs, ys)
plt.scatter(x_predict, y_predict, s=100)
plt.plot(xs, regression_line)
plt.show()

print(m, b)
