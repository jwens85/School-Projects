def linear_search(numbers, key):
  i = 0
  numbers_size = len(numbers)  # Get the size of the numbers list

  # Loop through the list
  while i < numbers_size:
      if numbers[i] == key:
          return i  # Return the index if the key is found
      i += 1

  return -1  # Return -1 if the key is not found

# Example usage:
numbers = [10, 23, 45, 70, 11, 15]
key = 45

# Perform the linear search
result = linear_search(numbers, key)

if result != -1:
  print(f"Key found at index {result}")  # Output will be "Key found at index 2"
else:
  print("Key not found")
