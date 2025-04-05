# Partition function used by the Quicksort algorithm
def partition(numbers, lowIndex, highIndex):
    # Pick the middle element as pivot
    midpoint = lowIndex + (highIndex - lowIndex) // 2
    pivot = numbers[midpoint]

    done = False
    while not done:
        # Increment lowIndex while numbers[lowIndex] < pivot
        while numbers[lowIndex] < pivot:
            lowIndex += 1

        # Decrement highIndex while pivot < numbers[highIndex]
        while pivot < numbers[highIndex]:
            highIndex -= 1

        # If lowIndex meets or crosses highIndex, partition is complete
        if lowIndex >= highIndex:
            done = True
        else:
            # Swap numbers[lowIndex] and numbers[highIndex]
            temp = numbers[lowIndex]
            numbers[lowIndex] = numbers[highIndex]
            numbers[highIndex] = temp

            # Update lowIndex and highIndex
            lowIndex += 1
            highIndex -= 1

    return highIndex  # Return the index of the last element in the low partition


# Quicksort algorithm that uses the partition function
def quicksort(numbers, lowIndex, highIndex):
    # Base case: if partition size is 1 or zero, it's already sorted
    if lowIndex >= highIndex:
        return

    # Partition the data and get the index of the last element in the low partition
    lowEndIndex = partition(numbers, lowIndex, highIndex)

    # Recursively sort the low and high partitions
    quicksort(numbers, lowIndex, lowEndIndex)
    quicksort(numbers, lowEndIndex + 1, highIndex)


# Main program to test the Quicksort algorithm
def main():
    # Initialize the list of numbers to sort
    numbers = [10, 2, 78, 4, 45, 32, 7, 11]
    NUMBERS_SIZE = len(numbers)

    # Print the unsorted array
    print("UNSORTED:")
    for i in range(NUMBERS_SIZE):
        print(numbers[i], end=" ")
    print()  # Print a newline

    # Initial call to quicksort
    quicksort(numbers, 0, NUMBERS_SIZE - 1)

    # Print the sorted array
    print("SORTED:")
    for i in range(NUMBERS_SIZE):
        print(numbers[i], end=" ")
    print()  # Print a newline


# Call the main function to execute the quicksort algorithm
main()

"""
Summary:
This program implements the Quicksort algorithm using the Partition function.

1. **Partition function**:
   - The pivot is selected as the middle element of the list.
   - The partitioning process rearranges the list so that elements less than the pivot are on the left, and elements greater than the pivot are on the right.
   - It returns the index of the last element in the lower partition, which helps divide the array for recursive sorting.

2. **Quicksort function**:
   - The Quicksort algorithm works by recursively partitioning the list into two sublists.
   - The base case occurs when a partition contains 1 or 0 elements, meaning it's already sorted.

The unsorted list [10, 2, 78, 4, 45, 32, 7, 11] is sorted using Quicksort, and the sorted list is printed at the end.
"""
