class MoveToFrontList:
  def __init__(self, items):
      self.items = items  # Initialize the list with given items

  def search(self, key):
      comparisons = 0  # To track the number of comparisons
      for i in range(len(self.items)):
          comparisons += 1  # Increment comparisons for each check
          if self.items[i] == key:
              # Move the found item to the front of the list
              found_item = self.items.pop(i)
              self.items.insert(0, found_item)
              return (i, comparisons)  # Return position and comparisons made
      return (-1, comparisons)  # If not found, return -1 and total comparisons

  def display(self):
      print("Current List:", self.items)

# Example usage:
if __name__ == "__main__":
  # Initial list of items
  initial_list = [17, 97, 9, 56, 65, 33, 64, 42]

  # Create the MoveToFrontList object
  move_to_front = MoveToFrontList(initial_list)

  # Display the original list
  print("Original List:")
  move_to_front.display()

  # Perform searches as per the example in the image
  print("\nSearch for 42:")
  index, comparisons = move_to_front.search(42)
  move_to_front.display()
  print(f"Index: {index}, Comparisons: {comparisons}")

  print("\nSearch for 42 again:")
  index, comparisons = move_to_front.search(42)
  move_to_front.display()
  print(f"Index: {index}, Comparisons: {comparisons}")

  print("\nSearch for 64:")
  index, comparisons = move_to_front.search(64)
  move_to_front.display()
  print(f"Index: {index}, Comparisons: {comparisons}")

  print("\nSearch for 42 again:")
  index, comparisons = move_to_front.search(42)
  move_to_front.display()
  print(f"Index: {index}, Comparisons: {comparisons}")

"""
Explanation:
============

This program implements the "Move-to-Front" self-adjusting heuristic. 

### Components:
1. **MoveToFrontList Class:**
 - This class contains the list of items and a method to search for a key.
 - The `search()` method performs a linear search through the list.
 - After finding the key, it moves that key to the front of the list using the list's `pop()` and `insert()` methods.
 - It also keeps track of how many comparisons were made during the search.

2. **Search Process:**
 - The `search()` method iterates over the list, checking each element for a match with the key.
 - If a match is found, the item is moved to the front, and the number of comparisons it took to find the item is returned.
 - If the key is not found, the method returns -1 and the total number of comparisons.

### How it Works (Step-by-Step Example):
1. **Initial List:** [17, 97, 9, 56, 65, 33, 64, 42]
 - Search for 42:
   - The linear search finds 42 at index 7 after 8 comparisons.
   - After finding 42, it is moved to the front of the list: [42, 17, 97, 9, 56, 65, 33, 64]

 - Search for 42 again:
   - Since 42 is now at the front of the list, it only takes 1 comparison.

 - Search for 64:
   - 64 is found at index 7 after 8 comparisons.
   - After finding 64, it is moved to the front: [64, 42, 17, 97, 9, 56, 65, 33]

 - Search for 42 again:
   - 42 is now at index 1, so it takes 2 comparisons to find it.

### Benefits of Move-to-Front Heuristic:
- Frequently searched elements move to the front, reducing the number of comparisons for future searches.
- In cases where some items are searched repeatedly, this heuristic reduces the overall number of comparisons.

"""

