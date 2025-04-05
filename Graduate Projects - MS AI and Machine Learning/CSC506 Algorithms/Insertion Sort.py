def insertion_sort(numbers):
    # Variable to track the number of cycles (shifts/swaps) made during the sorting process
    cycles = 0

    # Loop through all elements starting from the second one
    for i in range(1, len(numbers)):
        j = i
        # Insert numbers[i] into the sorted portion on the left
        # Continue shifting elements until the current number is in the correct position
        while j > 0 and numbers[j] < numbers[j - 1]:
            # Swap numbers[j] and numbers[j - 1] to move the current element leftward
            temp = numbers[j]
            numbers[j] = numbers[j - 1]
            numbers[j - 1] = temp
            j -= 1

            # Increment the cycles counter for each shift/swapping operation
            cycles += 1

    return cycles


# Main program to test the insertion sort algorithm
numbers = [10, 2, 78, 4, 45, 32, 7, 11]

# Display the unsorted array
print('UNSORTED:', numbers)

# Perform insertion sort and store the number of cycles
cycles = insertion_sort(numbers)

# Display the sorted array
print('SORTED:', numbers)

# Print the number of cycles taken to sort the array
print('Total cycles (swaps/iterations):', cycles)

"""
Summary:
This program implements the Insertion Sort algorithm and counts the number of cycles (swaps/iterations) it takes to sort the array. 

1. The algorithm moves each element in the unsorted part of the list leftward into its correct position within the sorted part by repeatedly shifting larger elements to the right.
2. A cycle is defined as a swap/iteration that occurs when an element is moved to the left.

In this example, the unsorted list [10, 2, 78, 4, 45, 32, 7, 11] is sorted, and the total number of cycles required is displayed at the end. 
"""
