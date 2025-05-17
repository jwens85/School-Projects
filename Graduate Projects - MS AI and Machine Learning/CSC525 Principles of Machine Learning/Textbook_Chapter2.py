import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# value = np.random.randint(0,1000000000)
# print(value)
# #------------------------------------------------
# few_rolls  = np.random.randint(1, 7, size=10)
# many_rolls = np.random.randint(1, 7, size=1000)
#
# print(few_rolls)
# print(many_rolls)
# #------------------------------------------------
# # Simulate rolling a fair six-sided die
# few_rolls = np.random.randint(1, 7, size=10)
# many_rolls = np.random.randint(1, 7, size=1000)
#
# # Count occurrences using histogram buckets from 0.5 to 7.5 (to center on integers 1–6)
# few_counts  = np.histogram(few_rolls,  bins=np.arange(0.5, 7.5))[0]
# many_counts = np.histogram(many_rolls, bins=np.arange(0.5, 7.5))[0]
#
# # Plot bar charts for comparison
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
#
# ax1.bar(np.arange(1, 7), few_counts, color='skyblue', edgecolor='black')
# ax1.set_title("Few Rolls (n=10)")
# ax1.set_xlabel("Die Face")
# ax1.set_ylabel("Count")
#
# ax2.bar(np.arange(1, 7), many_counts, color='steelblue', edgecolor='black')
# ax2.set_title("Many Rolls (n=1000)")
# ax2.set_xlabel("Die Face")
# ax2.set_ylabel("Count")
#
# plt.tight_layout()
# plt.show()
# #------------------------------------------------
# # Use the binomial distribution
# b = ss.distributions.binom
#
# # Plot how the distribution of heads changes as we increase the number of flips
# fig, axes = plt.subplots(1, 5, figsize=(15, 3))
#
# for ax, flips in zip(axes, [5, 10, 20, 40, 80]):
#     x = np.arange(flips + 1)  # possible number of heads: 0 to flips
#     y = b.pmf(x, flips, 0.5)  # probability mass function with p=0.5 (fair coin)
#
#     ax.bar(x, y, color='steelblue', edgecolor='black')
#     ax.set_title(f'{flips} Flips')
#     ax.set_xlabel('Number of Heads')
#     ax.set_ylabel('Probability')
#     ax.set_ylim(0, 0.3)
#
# plt.tight_layout()
# plt.show()
# #------------------------------------------------
# b = ss.distributions.binom
# for flips in [5, 10, 20, 40, 80]:
#     success = np.arange(flips)
#     our_distribution = b.pmf(success, flips, .5)
#     plt.hist(success, flips, weights=our_distribution)
# plt.xlim(0, 55);
# plt.show()
# #------------------------------------------------
# # Define binomial and normal distribution objects
# b = ss.distributions.binom
# n = ss.distributions.norm
#
# # Set up the figure
# plt.figure(figsize=(10, 6))
#
# for flips in [5, 10, 20, 40, 80]:
#     # Binomial distribution for flipping `flips` coins
#     success = np.arange(flips + 1)  # include flips itself
#     our_distribution = b.pmf(success, flips, 0.5)
#     plt.hist(success, bins=flips, weights=our_distribution, alpha=0.5,
#              label=f'{flips} flips - binomial')
#
#     # Normal approximation
#     mu = flips * 0.5
#     std_dev = np.sqrt(flips * 0.5 * 0.5)
#
#     norm_x = np.linspace(mu - 3 * std_dev, mu + 3 * std_dev, 200)
#     norm_y = n.pdf(norm_x, loc=mu, scale=std_dev)
#     plt.plot(norm_x, norm_y, label=f'{flips} flips - normal', linewidth=1.5)
#
# # Styling
# plt.title("Binomial Distribution vs. Normal Approximation")
# plt.xlabel("Number of Heads")
# plt.ylabel("Probability")
# plt.xlim(0, 55)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# #------------------------------------------------
# # pure python, old-school
# quantity = [2, 12, 3]
# costs = [12.5, .5, 1.75]
# partial_cost = []
# for q,c in zip(quantity, costs):
#     partial_cost.append(q*c)
# sum(partial_cost)
# print("Partial Costs are", partial_cost)
# print("Total Costs are", sum(costs))
# #------------------------------------------------
# pure python, for the new-school, cool kids
# quantity = np.array([2, 12, 3])
# costs = np.array([12.5, 0.5, 1.75])
#
# print(
#     quantity.dot(costs),      # Dot-product way 1
#     np.dot(quantity, costs),  # Dot-product way 2
#     quantity @ costs,         # Dot-product way 3 (Python 3.5+)
#     sep='\n'
# )
# #------------------------------------------------
# quantity = np.array([2, 12, 3])
# costs = np.array([12.5, 0.5, 1.75])
#
# for q_i, c_i in zip(quantity, costs):
#     print("{:2d} {:5.2f} --> {:5.2f}".format(int(q_i), float(c_i), q_i * c_i))
#
# print("Total:",
#       sum(q * c for q, c in zip(quantity, costs)))  # Cool-kid method
# #------------------------------------------------
# values = np.array([10.0, 20.0, 30.0]) # Input values
# weights = np.full_like(values, 1/3) # Equal weights: [1/3, 1/3, 1/3]
#
# print("weights:", weights)
# print("via mean:", np.mean(values)) # Regular mean
# print("via weights and dot:", np.dot(weights, values)) # Weighted sum via dot product
# #------------------------------------------------
# # Values and weights from the example
# values = np.array([10, 20, 30])
# weights = np.array([0.5, 0.25, 0.25])
#
# # Compute and print the weighted average using dot product
# weighted_average = np.dot(weights, values)
# print("Weighted Average:", weighted_average)
# #------------------------------------------------
# # Define payoffs: +$1.00 for odd, -$0.50 for even
# payoffs = np.array([1.0, -0.5])    # [odd, even]
# # Define probabilities: 50% chance for odd, 50% chance for even
# probs = np.array([0.5, 0.5])
# # Compute expected value using dot product
# expected_value = np.dot(payoffs, probs)
# # Print the result
# print("Expected Value:", expected_value)
# #------------------------------------------------
# def is_even(n):
#     # if remainder 0, value is even
#     return n % 2 == 0
# winnings = 0.0
# for toss_ct in range(10000):
#     die_toss = np.random.randint(1, 7)
#     winnings += 1.0 if is_even(die_toss) else -0.5
# print(winnings)
# #------------------------------------------------
# #Sum of squares
# values = np.array([5, -3, 2, 1])
# # Element-wise squaring
# squares = values * values
# # Print the individual squares
# print(squares)
# # Method 1: Sum of element-wise squares
# print(np.sum(squares))
# # Method 2: Dot product of the vector with itself
# print(np.dot(values, values))
# #------------------------------------------------
# #Sum of Errors
# # Simulated prediction errors
# errors = np.array([5, -5, 3.2, -1.1])
# # Create a DataFrame with errors and squared errors
# df = pd.DataFrame({
#     'errors': errors,
#     'squared': errors * errors
# })
# # Print the table and total squared error
# print(df)
# print("\nTotal Squared Error:", np.sum(errors * errors))
# #------------------------------------------------
# # Define the number of people (1 to 10)
# people = np.arange(1, 11)
# # Parking cost is fixed at $40, regardless of how many people go
# total_cost = np.ones_like(people) * 40.0
# # Plotting
# fig, ax = plt.subplots()
# ax.plot(people, total_cost)
# # Labels
# ax.set_xlabel("# People")
# ax.set_ylabel("Cost\n(Parking Only)")
# # Display the plot
# plt.show()
# #------------------------------------------------
# # Number of people (1 to 10)
# people = np.arange(1, 11)
# # Calculate total cost: $80 per ticket + $40 flat parking fee
# total_cost = 80.0 * people + 40.0
# # Optional: display table (transpose to save vertical space)
# df = pd.DataFrame({'total_cost': total_cost.astype(int)}, index=people).T
# print(df)
# # Plotting
# fig, ax = plt.subplots()
# ax.plot(people, total_cost, 'bo')  # 'bo' means blue circles
# ax.set_ylabel("Total Cost")
# ax.set_xlabel("People")
# plt.show()
# #------------------------------------------------
# # Generate 100 x values between -3 and 3
# xs = np.linspace(-3, 3, 100)
#
# # Define slope and intercept
# m, b = 1.5, -3
#
# # Create the figure and axis
# fig, ax = plt.subplots()
#
# # Calculate y = mx + b
# ys = m * xs + b
# ax.plot(xs, ys, label='y = 1.5x - 3')  # main line in blue
#
# # Set y-axis range
# ax.set_ylim(-4, 4)
#
# # Plot red points: y-intercept and a second point two steps to the right
# ax.plot(0, b, 'ro')       # y-intercept at (0, -3)
# ax.plot(2, m*2 + b, 'ro') # (2, 0)
#
# # Plot horizontal line (slope m = 0) for comparison
# ys_flat = 0 * xs + b
# ax.plot(xs, ys_flat, 'y', label='y = -3')
#
# # Optional: make axis look like high school graph paper
# ax.spines['left'].set_position(('data', 0))
# ax.spines['bottom'].set_position(('data', 0))
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_aspect('equal')
# ax.grid(True)
#
# # Labels and legend
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.legend()
#
# # Show the plot
# plt.show()
# #------------------------------------------------
# # Step 1: Generate data and add a column of ones (plus-one trick)
# xs = np.linspace(-3, 3, 100)                        # 100 points from -3 to 3
# xs_p1 = np.c_[xs, np.ones_like(xs)]                # shape: (100, 2)
#
# # Step 2: Define slope and intercept as a vector
# w = np.array([1.5, -3])                             # slope = 1.5, intercept = -3
#
# # Step 3: Compute ys = dot(xs_p1, w)
# ys = np.dot(xs_p1, w)
#
# # Step 4: Plot the line
# fig, ax = plt.subplots()
# ax.plot(xs, ys)                                     # blue line
# ax.set_ylim(-4, 4)
#
# # Step 5: Add red points for interpretation
# ax.plot(0, -3, 'ro')                                # y-intercept (0, -3)
# ax.plot(2, 0, 'ro')                                 # (2, 0) after applying y = 1.5x - 3
#
# # Optional styling to make it look like a math graph
# ax.spines['left'].set_position(('data', 0))
# ax.spines['bottom'].set_position(('data', 0))
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_aspect('equal')
# ax.grid(True)
#
# plt.show()
# #------------------------------------------------
# # Create input ranges
# number_people = np.arange(1, 11)      # 1 to 10 people
# number_rbs = np.arange(0, 20)         # 0 to 19 rootbeers
#
# # Create 2D grids from these 1D ranges
# number_people, number_rbs = np.meshgrid(number_people, number_rbs)
#
# # Compute total cost on the grid
# total_cost = 80 * number_people + 10 * number_rbs + 40
#
# # Create subplots with 3D projection
# fig, axes = plt.subplots(2, 3, subplot_kw={'projection': '3d'}, figsize=(9, 6))
#
# # List of different viewing angles (azimuths)
# angles = [0, 45, 90, 135, 180]
#
# # Plot the surface from each angle
# for ax, angle in zip(axes.flat, angles):
#     ax.plot_surface(number_people, number_rbs, total_cost, cmap='viridis')
#     ax.set_xlabel("People")
#     ax.set_ylabel("RootBeers")
#     ax.set_zlabel("TotalCost")
#     ax.azim = angle  # change the azimuth angle
#
# # Turn off the last empty plot (6th subplot)
# axes.flat[-1].axis('off')
#
# # Improve spacing
# fig.tight_layout()
# plt.show()
