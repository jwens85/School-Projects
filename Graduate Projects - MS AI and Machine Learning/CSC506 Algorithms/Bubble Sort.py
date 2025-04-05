def bubble_sort(data):
    # set the variable n to the length of the list of data
    n = len(data)

    # Initialize a cycle counter
    cycles = 0

    # The outer loop will iterate n-1 times
    for i in range(n):
        # Initialize the swapped flag for the inner loop's iterations, right now, swapped is set to false
        swapped = False

        # The inner loop compares adjacent values and swaps them if out of order
        # Each iteration the range decreases as sorted values 'bubble' up
        for j in range(0, n - i - 1):
            cycles += 1  # The cycle counter increments with each comparison

            # If the current value is greater than the next value in the list, the values are swapped
            if data[j] > data[j + 1]:
                # Swapping values
                data[j], data[j + 1] = data[j + 1], data[j]
                # And mark that a swap has occured
                swapped = True

        # If no values are swapped, the list is sorted
        if not swapped:
            break  #The outer loop is exited

    # Print the sorted data, and the number of cycles required to sort the list
    return data, cycles


# Example Dataset
data = [64, 18, 34, 19, 25, 19, 12, 29, 5, 11, 90]

# Print the name of the algorithm
print("Bubble Sort")

# Print the unsorted list
print("UNSORTED:", data)

# Call the bubble_sort function and get the sorted list along with number of cycles
sorted_data, total_cycles = bubble_sort(data)

# Print the sorted list
print("SORTED:", sorted_data)

# Print the total number of cycles
print("Total cycles (comparisons):", total_cycles)

"""

Summary:
This is a basic bubble sort function that sorts a list of numbers and count the number of cycles required to sort the list

1. Each comparison on the inner loop is counted as a cycle
2. Prints the sorted list and total cycles required to sort
3. The cycle counter is useful to compare time complexity against other algorithms

"""
