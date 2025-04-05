# Function to perform a binary search on a sorted list of product names
def binary_search(products, products_size, key):
    low = 0  # Initialize the low index
    high = products_size - 1  # Initialize the high index

    # Continue searching while the range is valid
    while high >= low:
        mid = (high + low) // 2  # Calculate the middle index

        # Compare the key with the product name at the mid position
        if products[mid] < key:
            low = mid + 1  # If key is greater, search in the upper half
        elif products[mid] > key:
            high = mid - 1  # If key is smaller, search in the lower half
        else:
            return mid  # If the key matches the mid product, return the index

    return -1  # Return -1 if the product is not found

# Main function to execute the logic
def main():
    # Initialize a sorted list of product names
    products = ["Apples", "Bananas", "Carrots", "Dates", "Eggs", "Grapes", "Oranges", "Peaches"]
    # Initialize a corresponding list of product numbers
    product_numbers = [2, 4, 7, 10, 11, 32, 45, 87]  

    NUM_PRODUCTS = len(products)  # Get the size of the products list

    # Print the list of products with their associated product numbers
    print("PRODUCTS:")
    for i in range(NUM_PRODUCTS):
        # Print each product along with its corresponding product number
        print(f"{products[i]} (Product Number: {product_numbers[i]})")
    print()  # New line for better formatting

    # Get user input for the product name to search
    key = input("Enter the name of the product to search: ")

    # Perform binary search to find the product
    product_index = binary_search(products, NUM_PRODUCTS, key)

    # Check if the product was found
    if product_index == -1:
        print(f"Product '{key}' was not found.")  # Print if the product wasn't found
    else:
        product_number = product_numbers[product_index]  # Get the corresponding product number
        print(f"Found '{key}' with product number {product_number} at index {product_index}.")

# Call the main function to run the program
if __name__ == "__main__":
    main()

# EXPLANATION:
#
# 1. This code implements a binary search algorithm that operates on a list of sorted product names.
#    It efficiently finds the index of a product by repeatedly dividing the list in half and narrowing 
#    the search to the correct half.
#
# 2. The `binary_search` function takes three arguments:
#    - `products`: the sorted list of product names.
#    - `products_size`: the total number of products (size of the list).
#    - `key`: the product name the user wants to search for.
#
# 3. The function initializes two pointers: `low` and `high` to mark the start and end of the search range.
#    It calculates the midpoint (`mid`) and checks if the key matches the middle product.
#    - If the `key` is larger than the middle product, the function continues searching in the upper half.
#    - If the `key` is smaller, it searches the lower half.
#    - If the `key` matches the middle product, the function returns the index.
#    - If the key is not found, the function returns `-1`.
#
# 4. The `main` function initializes two lists: `products` (the product names) and `product_numbers`
#    (corresponding product numbers). It prints all the products, then prompts the user to input a product name.
#    The `binary_search` function is called to search for the product, and the program outputs the result,
#    either displaying the product's index and number or stating that the product was not found.
#
# 5. The code uses integer division (`//`) to compute the midpoint of the list, ensuring the index remains an integer.
# 6. This program provides an efficient O(log n) search mechanism, making it suitable for sorted datasets where
#    a more performant search is needed compared to linear search (O(n)).
