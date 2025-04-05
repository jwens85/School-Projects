def find_max(input_array):
    # Assume the first element is the largest initially
    max_value = input_array[0]

    # Iterate over the rest of the array starting from the second element
    for i in range(1, len(input_array)):
        if input_array[i] > max_value:
            max_value = input_array[i]

    return max_value

# Example usage:
arr = [3, 5, 2, 9, 6]
result = find_max(arr)  # The function will return 9
print("The maximum value is:", result)
