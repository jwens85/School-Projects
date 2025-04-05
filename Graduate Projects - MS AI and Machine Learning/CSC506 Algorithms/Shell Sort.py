def insertion_sort_interleaved(numbers, start_index, gap):
  # Variable to count the number of swaps in the interleaved sequence
  swaps = 0
  # Loop through the interleaved elements starting from start_index + gap
  for i in range(start_index + gap, len(numbers), gap):
      j = i
      # Perform insertion sort by comparing the current element with the one at gap distance
      while (j - gap >= start_index) and (numbers[j] < numbers[j - gap]):
          # Increment the swaps counter every time a swap is made
          swaps += 1
          # Swap numbers[j] and numbers[j - gap]
          temp = numbers[j]
          numbers[j] = numbers[j - gap]
          numbers[j - gap] = temp
          # Move j leftward by gap
          j = j - gap
  return swaps  # Return the total number of swaps made


def shell_sort(numbers, gap_values):
  # List to store the number of swaps made for each gap value
  swaps = []
  # Loop through the provided gap values
  for gap_value in gap_values:
      # Apply the interleaved insertion sort for each starting index based on the current gap
      for i in range(gap_value):
          swaps.append(insertion_sort_interleaved(numbers, i, gap_value))
  return swaps  # Return the total number of swaps for all gaps


# Main program to test the shell sort algorithm.
numbers = [12, 18, 3, 72, 65, 22, 19]
print('UNSORTED:', numbers)

# Perform shell sort with gap values [4, 2, 1] (classic Shell Sort)
swaps = shell_sort(numbers, [4, 2, 1])

# Display the sorted array
print('SORTED:', numbers)

# Display the total swaps made for each gap
print('Total swaps for each gap:', swaps)

"""
Summary:
This program implements the Shell Sort algorithm using the `insertion_sort_interleaved` function, which performs insertion sort on elements that are gap positions apart.

1. **Shell Sort:** The `shell_sort` function takes the list `numbers` and a list of `gap_values` (like [4, 2, 1]).
2. For each gap value, the array is sorted using interleaved insertion sort, where elements that are `gap` positions apart are compared and sorted.
3. The number of swaps made during each pass (for each gap value) is counted and stored in a list `swaps`.

In the example, the unsorted list `[12, 18, 3, 72, 65, 22, 19]` is sorted using gap values `[4, 2, 1]`. 
The sorted list and the total number of swaps for each gap are displayed.

**Explanation of Swap Counts:**
- Each gap value creates a sequence of elements that are sorted.
- The `swaps` list contains the total number of swaps made for each pass with a specific gap value.
"""
