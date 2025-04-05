from operator import attrgetter

# Class to represent an item with a weight and value
class Item:
    def __init__(self, item_weight, item_value):
        self.weight = item_weight  # Weight of the item
        self.value = item_value    # Value of the item

# Class to represent the knapsack
class Knapsack:
    def __init__(self, weight, items):
        self.max_weight = weight       # Maximum weight the knapsack can hold
        self.item_list = items         # List of items in the knapsack

# Function to implement the 0/1 knapsack using a greedy approach
def knapsack_01(knapsack, item_list):
    # Sort the items in descending order based on value
    item_list.sort(key = attrgetter('value'), reverse = True)

    # Track the remaining weight capacity in the knapsack
    remaining = knapsack.max_weight

    # Iterate through the sorted list of items
    for item in item_list:
        # If the item fits in the knapsack, add it
        if item.weight <= remaining:
            knapsack.item_list.append(item)
            remaining = remaining - item.weight  # Update the remaining weight

# Main program
# Create some items
item_1 = Item(6, 25)
item_2 = Item(8, 42)
item_3 = Item(12, 60)
item_4 = Item(18, 95)
item_list = [item_1, item_2, item_3, item_4]  # List of items
initial_knapsack_list = []  # Initially, the knapsack is empty

# Get the maximum weight the knapsack can hold from the user
max_weight = int(input('Enter maximum weight the knapsack can hold: '))

# Create a Knapsack object
knapsack = Knapsack(max_weight, initial_knapsack_list)

# Perform the knapsack algorithm
knapsack_01(knapsack, item_list)

# Output the items in the knapsack
print('Objects in knapsack')
i = 1
sum_weight = 0
sum_value = 0

# Print details of each item in the knapsack
for item in knapsack.item_list:
    sum_weight += item.weight
    sum_value += item.value
    print('%d: weight %d, value %d' % (i, item.weight, item.value))
    i += 1

print()
print('Total weight of items in knapsack: %d' % sum_weight)
print('Total value of items in knapsack: %d' % sum_value)

"""
Explanation:
============

This program is an implementation of a greedy version of the 0/1 knapsack problem using object-oriented principles.

### Components:
1. **Item Class:**
   - Represents individual items that have two properties: `weight` and `value`.
   - These are essential because each item has a weight (how much space it takes in the knapsack) and a value (how much it's worth).

2. **Knapsack Class:**
   - This class represents the knapsack that will hold the items. 
   - It contains a `max_weight`, which defines the weight limit of the knapsack, and an `item_list`, which holds the items currently in the knapsack.

3. **Knapsack Algorithm:**
   - The function `knapsack_01` implements a greedy algorithm that sorts the items based on their value in descending order.
   - After sorting, the function tries to add as many items as possible to the knapsack without exceeding the maximum weight.
   - It uses a `remaining` variable to track the available weight capacity left in the knapsack.

### Steps of the Algorithm:
1. **Input the Maximum Weight:** 
   - The user is prompted to input the maximum weight the knapsack can hold.

2. **Sort Items by Value:**
   - The items are sorted in descending order by their value using Python's `attrgetter` function. Sorting by value prioritizes higher-value items first.

3. **Adding Items to the Knapsack:**
   - The algorithm iterates through the sorted list of items and checks if each item fits in the remaining weight capacity.
   - If the item fits, it is added to the knapsack, and the remaining weight capacity is updated.

4. **Output the Results:**
   - After the algorithm runs, it prints the weight and value of each item that was added to the knapsack.
   - It also prints the total weight and total value of the items in the knapsack.

### Example Walkthrough:
- Suppose you have items with the following weights and values: 
  - Item 1: weight 6, value 25
  - Item 2: weight 8, value 42
  - Item 3: weight 12, value 60
  - Item 4: weight 18, value 95
- The algorithm will sort the items based on value:
  - Item 4 (value 95), Item 3 (value 60), Item 2 (value 42), Item 1 (value 25)
- If the user enters a maximum weight of 20:
  - The algorithm will first add Item 4 (weight 18, value 95) since it has the highest value.
  - The remaining capacity is 2, and no more items can fit.
  - The total weight in the knapsack is 18, and the total value is 95.

### Limitations:
- **Greedy Approach:** This algorithm sorts items by value and adds them in descending order, but it may not always give the optimal solution for the 0/1 knapsack problem.
- For an optimal solution, a dynamic programming approach would be required to explore all possible combinations of items.

"""
