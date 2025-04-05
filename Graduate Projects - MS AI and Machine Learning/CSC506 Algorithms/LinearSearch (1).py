# Function to perform a linear search on a list
def linear_search(numbers, numbers_size, key):
    # Loop through each element in the list starting from index 0
    for i in range(numbers_size):
        # Check if the current element is equal to the key
        if numbers[i] == key:
            return i  # Return the index if the key is found
    return -1  # Return -1 if the key is not found after checking all elements

# Main function to execute the logic
def main():
    # Initialize the list of numbers
    numbers = [2, 4, 7, 10, 11, 32, 45, 87]

    # Get the size of the numbers list
    NUMBERS_SIZE = len(numbers)

    # Print the list of numbers
    print("NUMBERS: ")
    for number in numbers:
        # Print each number followed by a space
        print(str(number) + " ", end='')
    print()  # Move to the next line after printing the numbers

    # Get user input for the number to search
    key = int(input("Enter a value: "))

    # Perform linear search to find the key
    key_index = linear_search(numbers, NUMBERS_SIZE, key)

    # Check if the key was found and print the appropriate message
    if key_index == -1:
        print(f"{key} was not found.")  # If key is not found
    else:
        print(f"Found {key} at index {key_index}.")  # If key is found, print the index

# Call the main function to run the program
if __name__ == "__main__":
    main()

# EXPLANATION:
#
# 1. The linear search algorithm is the simplest search algorithm that sequentially checks each element
#    in a list until it finds the key or reaches the end of the list.
#
# 2. The `linear_search` function takes three parameters:
#    - `numbers`: A list of numbers to search through.
#    - `numbers_size`: The size of the list (or the number of elements in the list).
#    - `key`: The value that the user is searching for in the list.
#
# 3. The function iterates through the list using a `for` loop. For each iteration, it compares the 
#    current element to the key:
#    - If the element matches the key, it returns the index of that element.
#    - If the function reaches the end of the list without finding the key, it returns `-1` to indicate
#      that the key was not found.
#
# 4. The `main` function initializes a list of numbers (`numbers`) and prints it out. It then prompts the
#    user to input a number to search for (`key`). 
#    - The program uses `linear_search` to find the key in the list.
#    - Depending on whether the key was found or not, the program prints a message stating whether the
#      number was found and its index or that the number was not found.
#
# 5. This algorithm has a time complexity of O(n), meaning the time it takes to find the key is proportional
#    to the number of elements in the list. While it's simple to implement, it's not the most efficient
#    method for searching in large datasets, especially when compared to algorithms like binary search.
