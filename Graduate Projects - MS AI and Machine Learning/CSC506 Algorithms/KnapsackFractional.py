from operator import attrgetter

# Class to represent an item with a weight and value
class Item:
    def __init__(self, item_weight, item_value):
        self.weight = item_weight  # Weight of the item
        self.value = item_value    # Value of the item

    # Method to compute the value-to-weight ratio
    def value_per_weight(self):
        return self.value / self.weight

# Class to represent the knapsack
class Knapsack:
    def __init__(self, weight):
        self.max_weight = weight       # Maximum weight the knapsack can hold
        self.total_value = 0           # Total value of items in the knapsack
        self.items_in_knapsack = []    # List of items (or fractions of items) in the knapsack

# Function to implement the fractional knapsack algorithm
def fractional_knapsack(knapsack, item_list):
    # Sort the items by value/weight ratio in descending order
    item_list.sort(key=lambda item: item.value_per_weight(), reverse=True)

    remaining_weight = knapsack.max_weight

    for item in item_list:
        # If the entire item can fit, add it to the knapsack
        if item.weight <= remaining_weight:
            knapsack.items_in_knapsack.append(item)
            knapsack.total_value += item.value
            remaining_weight -= item.weight
        else:
            # Add a fraction of the item that can fit into the remaining space
            fraction = remaining_weight / item.weight
            knapsack.total_value += item.value * fraction
            # Store the fraction of the item that was added
            knapsack.items_in_knapsack.append(f"Fraction of item ({fraction * 100:.2f}%) with value {item.value * fraction:.2f}")
            break  # Stop since the knapsack is now full

    return knapsack

# Main program
# Create some items
item_1 = Item(10, 60)  # Item with weight 10 and value 60
item_2 = Item(20, 100)  # Item with weight 20 and value 100
item_3 = Item(30, 120)  # Item with weight 30 and value 120
item_list = [item_1, item_2, item_3]

# Get the maximum weight the knapsack can hold
max_weight = int(input("Enter the maximum weight the knapsack can hold: "))

# Create a Knapsack object
knapsack = Knapsack(max_weight)

# Perform the fractional knapsack algorithm
knapsack = fractional_knapsack(knapsack, item_list)

# Output the results
print("Items in the knapsack:")
for item in knapsack.items_in_knapsack:
    if isinstance(item, Item):
        print(f"Full item with weight {item.weight} and value {item.value}")
    else:
        print(item)  # Print fraction of item

print(f"\nTotal value of items in the knapsack: {knapsack.total_value:.2f}")

"""
Explanation:
============

This program implements the **Fractional Knapsack** problem using a greedy algorithm. In this problem, we can take fractional parts of items to maximize the total value of the items in the knapsack while staying within a given weight limit.

### Components:
1. **Item Class:**
   - Each `Item` object has two properties: `weight` and `value`.
   - A method `value_per_weight()` calculates the ratio of value to weight for each item, which is used to determine how desirable an item is (higher value-to-weight ratio is better).

2. **Knapsack Class:**
   - This class represents the knapsack that will hold the items. 
   - It contains a `max_weight` property that defines the maximum capacity of the knapsack.
   - It also tracks the `total_value` of items in the knapsack and stores the list of `items_in_knapsack`.

3. **Fractional Knapsack Algorithm:**
   - The function `fractional_knapsack()` implements the greedy algorithm for solving the fractional knapsack problem.
   - The items are first sorted by their value-to-weight ratio in descending order using Python’s `sort()` function.
   - Then, items are added to the knapsack:
     - If the item can fit fully, it is added to the knapsack, and its value is added to the total value.
     - If the item cannot fit fully, a fraction of the item is added, and the corresponding fraction of the value is calculated and added to the total value.
     - The loop terminates when the knapsack is full.

### Example Walkthrough:
- Let's say the user inputs a maximum knapsack weight of 50, and the available items are:
  - Item 1: weight 10, value 60
  - Item 2: weight 20, value 100
  - Item 3: weight 30, value 120
- The algorithm first calculates the value-to-weight ratios:
  - Item 1: 60 / 10 = 6
  - Item 2: 100 / 20 = 5
  - Item 3: 120 / 30 = 4
- It then sorts the items by this ratio, resulting in the same order (Item 1, Item 2, Item 3).
- It adds Item 1 fully (weight 10, value 60) and Item 2 fully (weight 20, value 100), leaving 20 units of capacity.
- For Item 3, only part of the item can fit, so it adds a fraction (20/30 = 0.67 or 66.67%) of the item's weight and value (0.67 * 120 = 80).
- The total value in the knapsack is 60 + 100 + 80 = 240.

### Output:
- The program outputs the full items added to the knapsack and the fractional part of any item that couldn't fully fit.
- The total value of all items in the knapsack is printed at the end.

### Limitations:
- This greedy algorithm gives the optimal solution for the fractional knapsack problem because taking fractions of items allows for fine-grained control over the weight added to the knapsack.
"""
