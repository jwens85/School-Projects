def partition(numbers, start_index, end_index):
  pivot = numbers[end_index]
  i = start_index - 1
  for j in range(start_index, end_index):
      if numbers[j] <= pivot:
          i += 1
          numbers[i], numbers[j] = numbers[j], numbers[i]
  numbers[i + 1], numbers[end_index] = numbers[end_index], numbers[i + 1]
  return i + 1

def quickselect(numbers, start_index, end_index, k):
  if start_index == end_index:
      return numbers[start_index]

  # Partition the array and get the pivot index
  pivot_index = partition(numbers, start_index, end_index)

  # Calculate the number of elements in the left partition
  if k == pivot_index:
      return numbers[pivot_index]
  elif k < pivot_index:
      return quickselect(numbers, start_index, pivot_index - 1, k)
  else:
      return quickselect(numbers, pivot_index + 1, end_index, k)
