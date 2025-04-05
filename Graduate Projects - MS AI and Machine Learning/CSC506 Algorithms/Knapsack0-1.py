def knapsack01(knapsack_max_weight, item_list, item_list_size):
  # Sort item list by descending value
  item_list.sort(key=lambda x: x['value'], reverse=True)

  # Remaining weight in the knapsack
  remaining_weight = knapsack_max_weight

  # Items that will be put into the knapsack
  knapsack_items = []

  # Iterate through the sorted item list
  for i in range(item_list_size):
      item = item_list[i]
      if item['weight'] <= remaining_weight:
          # Add item to the knapsack
          knapsack_items.append(item)
          # Reduce the remaining weight
          remaining_weight -= item['weight']

  return knapsack_items

# Example usage:
items = [
  {'name': 'item1', 'weight': 10, 'value': 60},
  {'name': 'item2', 'weight': 20, 'value': 100},
  {'name': 'item3', 'weight': 30, 'value': 120}
]

knapsack_max_weight = 50
item_list_size = len(items)

selected_items = knapsack01(knapsack_max_weight, items, item_list_size)

# Output selected items
for item in selected_items:
  print(f"Selected: {item['name']} (Weight: {item['weight']}, Value: {item['value']})")

"""
Explanation:
============

This program solves a variant of the 0/1 Knapsack Problem using a greedy algorithm.
It selects items based on their value (highest first) until the knapsack is full.

### Components:
1. **Item List:** 
 The items are represented as dictionaries with keys for the item's name, weight, and value.
 Example: {'name': 'item1', 'weight': 10, 'value': 60}

2. **Knapsack Maximum Weight:**
 The knapsack has a weight limit, defined by `knapsack_max_weight`. The goal is to fill the knapsack
 with items that maximize the total value without exceeding this weight limit.

### Steps of the Algorithm:
1. **Sorting Items by Value:**
 - The first thing we do is sort the item list in **descending order by value**.
 - This ensures that items with the highest value are considered first. 
 - Sorting is done using the `sort()` function with a lambda function to extract the value from each item.
   The sorting key: `lambda x: x['value']` means that the sort will consider the 'value' field of each item.

2. **Greedy Selection of Items:**
 - After sorting, we iterate over each item. If an item's weight is less than or equal to the 
   remaining weight capacity of the knapsack, it is added to the knapsack.
 - We also update the remaining weight of the knapsack by subtracting the weight of the newly added item.

3. **Termination:**
 - The algorithm stops either when all items are processed or when the knapsack's remaining weight is too small
   to accommodate any more items.

### Limitations:
- This greedy approach does not guarantee the optimal solution in all cases. In some situations, a different combination 
of items may lead to a higher total value, even if it means skipping over higher-value items.
- For a true solution to the 0/1 Knapsack Problem, you'd typically use **dynamic programming**. However, this greedy method 
is a fast, simple approximation that works well in some cases.

### Example:
- Given the list of items: [{'name': 'item1', 'weight': 10, 'value': 60}, {'name': 'item2', 'weight': 20, 'value': 100}, 
{'name': 'item3', 'weight': 30, 'value': 120}], and a knapsack with a maximum weight of 50:
- The items will first be sorted by their value, so the algorithm will check 'item3' (value: 120, weight: 30) first.
- Since 'item3' fits into the knapsack (weight 30 <= 50), it is added.
- The remaining weight in the knapsack is now 20 (50 - 30).
- Next, the algorithm checks 'item2' (value: 100, weight: 20), which also fits into the remaining capacity.
- 'item2' is added, and the knapsack is now full with a total weight of 50.
- 'item1' (value: 60, weight: 10) is not considered because the knapsack is already full.

The output will be:
Selected: item3 (Weight: 30, Value: 120)
Selected: item2 (Weight: 20, Value: 100)
"""
