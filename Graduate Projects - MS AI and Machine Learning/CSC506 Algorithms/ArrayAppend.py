def array_append(array, new_item):
  # Get the current size of the array
  current_size = len(array)

  # Simulate increasing array size (in Python, lists are dynamic so this isn't necessary)
  # array.append(new_item) would be the easy way, but we'll manually resize
  array += [None]  # Manually increasing size by appending a None

  # Insert the new item at the end of the array
  array[current_size] = new_item

# Example usage:
arr = [1, 2, 3]
new_item = 4
array_append(arr, new_item)
print("Array after append:", arr)  # Output will be [1, 2, 3, 4]
