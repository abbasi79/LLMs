import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 1. Generate sample data
np.random.seed(42)  # for reproducibility
x = np.array(lag3)
mode_result = stats.mode(lag3)

# y_mean = 2 * x + 1  # Example: a linear trend for the mean
y_mean = np.mean(x)  # Example: a linear trend for the mean
y_mean=np.repeat(y_mean,400,axis=0)

y_data = y_mean + np.random.normal(0, 1.5, size=len(x))  # Add noise
y_coords = np.arange(len(x))
y_data=y_coords
# 2. Calculate the standard deviation (or a measure of spread)
# For simplicity, let's assume a constant standard deviation for this example.
# In a real scenario, you might calculate it based on multiple observations per x-value
# or from a statistical model.
std_dev = 1.5

# 3. Create the scatter plot
plt.scatter(y_data,x, label='Data Points', alpha=0.7)

# 4. Calculate the upper and lower bounds for the shaded region
# Here, we're shading +/- 1 standard deviation from the mean trend.
y_upper = y_mean + std_dev
y_lower = y_mean - std_dev

# 5. Add the shaded region using fill_between
# plt.fill_between(x, y_lower, y_upper, color='blue', alpha=0.2, label='±1 Std Dev')
plt.fill_between(x, y_lower, y_upper, color='blue', alpha=0.2, label='±1 Std Dev')


# 6. Add the mean line (optional, but good for context)
# plt.plot(x, y_mean, color='red', linestyle='--', label='Mean Trend')
plt.plot(y_mean, x,color='red', linestyle='--', label='Mean Trend')


# 7. Customize the plot
plt.title('Scatter Plot with Shaded Standard Deviation')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()

