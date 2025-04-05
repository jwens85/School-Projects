# Define the merge function with the arguments numbers, i, j, k, and cycles
def merge(numbers, i, j, k, cycles):
    # The size of the first partition 
    mergedSize = k - i + 1
    # The position where merged numbers will be placed
    mergePos = 0
    # Define the variables leftPOS and rightPOS by their place in the partition
    leftPos = i
    rightPos = j + 1
    # Create a temporary array to store the merged numbers outside of the list itself
    mergedNumbers = [0] * mergedSize

    # A while loop that is setup to run when the left partition is less than or equal to the right partition
    while leftPos <= j and rightPos <= k:
        # The cycle counter increments with each comparison
        cycles[0] += 1
        if numbers[leftPos] <= numbers[rightPos]:
            mergedNumbers[mergePos] = numbers[leftPos]
            leftPos += 1
        else:
            mergedNumbers[mergePos] = numbers[rightPos]
            rightPos += 1
        mergePos += 1

    # Add any remaining elements in the left partition
    while leftPos <= j:
        mergedNumbers[mergePos] = numbers[leftPos]
        leftPos += 1
        mergePos += 1

    # Add any remaining elements in the right partition
    while rightPos <= k:
        mergedNumbers[mergePos] = numbers[rightPos]
        rightPos += 1
        mergePos += 1

    # Place the merged elements back into the original list
    for mergePos in range(mergedSize):
        numbers[i + mergePos] = mergedNumbers[mergePos]


# Define the merge_sort function with the arguments numbers, i, k, and cycles
def merge_sort(numbers, i, k, cycles):
    if i < k:
        # Find the partition's midpoint using the // operator
        j = (i + k) // 2

        # Use recursion to sort both partitions
        merge_sort(numbers, i, j, cycles)
        merge_sort(numbers, j + 1, k, cycles)

        # Merge the partitions and count the number of cycles
        merge(numbers, i, j, k, cycles)


# Call the main program to do a test of the code
def main():
    # Test list
    numbers = [64, 18, 34, 19, 25, 19, 12, 29, 5, 11, 90]
    NUMBERS_SIZE = len(numbers)

    #Print the name of the algorithm
    print("Merge Sort")

    # Print the unsorted list
    print("UNSORTED:")
    for i in range(NUMBERS_SIZE):
        print(numbers[i], end=" ")
    print()  # Start a new line

    # Define the variable 'cycles' to act as a counter
    cycles = [0]

    # Call the merge_sort function to sort the list
    merge_sort(numbers, 0, NUMBERS_SIZE - 1, cycles)

    # Print the sorted list
    print("SORTED:")
    for i in range(NUMBERS_SIZE):
        print(numbers[i], end=" ")
    print()  # Start a new line

    # Print the cycle count
    print("Total cycles (comparisons):", cycles[0])


# Call the main function to start the program
main()

"""
Summary:
This is a basic merge sort algorithm that sorts a list of numbers into ascending order, and keep track of the number of cycles required to sort the list.

1. Merge()
   - Sort the left and right partitions into a single list
   - Increment the cycle counter for each comparison

2. MergeSort()
   - Using the // operator, recursively split the list into two partitions
   - After the partitions are sorted, the Merge() function zips them back together

"""
