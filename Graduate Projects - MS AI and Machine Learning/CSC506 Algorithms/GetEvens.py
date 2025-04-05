def get_evens(lst, lst_size):
  i = 0
  evens_list = []  # Create a new, empty list to store even numbers
  while i < lst_size:
      if lst[i] % 2 == 0:  # Check if the current element is even
          evens_list.append(lst[i])  # Add the even number to evens_list
      i += 1
  return evens_list  # Return the list of even numbers

# Example usage:
lst = [3, 4, 7, 8, 10, 13]
lst_size = len(lst)

result = get_evens(lst, lst_size)
print("Even numbers in the list are:", result)  # Output: [4, 8, 10]
