def selection_sort(numbers):
    # Variable to track the number of cycles (swaps) made during the sorting process
    cycles = 0

    # Outer loop: Traverse the entire array
    for i in range(len(numbers) - 1):
        # Assume the current element is the smallest
        index_smallest = i

        # Inner loop: Find the smallest element in the remaining unsorted part of the list
        for j in range(i + 1, len(numbers)):
            if numbers[j] < numbers[index_smallest]:
                index_smallest = j

        # If the smallest element found is not at index i, swap it with numbers[i]
        if index_smallest != i:
            # Swap the elements
            temp = numbers[i]
            numbers[i] = numbers[index_smallest]
            numbers[index_smallest] = temp

            # Increment the cycles counter (a swap counts as a cycle)
            cycles += 1

    return cycles


# Main program to test the selection sort algorithm
numbers = [10, 2, 78, 4, 45, 32, 7, 11]

# Display the unsorted array
print('UNSORTED:', numbers)

# Perform selection sort and count the cycles
cycles = selection_sort(numbers)

# Display the sorted array
print('SORTED:', numbers)

# Display the total number of cycles taken to sort the array
print('Total cycles (swaps):', cycles)

"""
Summary:
This program implements the Selection Sort algorithm and tracks the number of cycles (swaps) performed during the sorting process. 

1. The algorithm finds the smallest element from the unsorted part of the list and swaps it with the first unsorted element.
2. A cycle is counted as each swap made to move the smallest element into its correct position.
3. The process continues until the list is fully sorted.

In this example, the unsorted list [10, 2, 78, 4, 45, 32, 7, 11] is sorted, and the total number of cycles required is printed at the end.
"""
