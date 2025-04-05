def longest_common_substring_optimized(str1, str2):
    # Step 1: Create a single row to store lengths of longest common suffixes
    # between substrings of str1 and str2.
    # We are using only one row to save memory, simulating the DP matrix.
    matrix_row = [0] * len(str2)

    # Step 2: Initialize variables to keep track of the maximum length of the
    # longest common substring found so far (max_value) and the row (max_value_row)
    # where this maximum value occurs in str1.
    max_value = 0
    max_value_row = 0

    # Step 3: Iterate through each character in str1 (outer loop for rows).
    for row in range(len(str1)):
        # Step 4: Initialize a variable to store the value from the upper-left diagonal
        # from the matrix (up_left), which is needed to simulate the diagonal value
        # in a full dynamic programming matrix.
        up_left = 0
        
        # Step 5: Iterate through each character in str2 (inner loop for columns).
        for col in range(len(str2)):
            # Save the current cell's value from matrix_row because this value
            # will be used as the diagonal (up-left) for the next iteration.
            saved_current = matrix_row[col]
        
            # Step 6: Check if the characters from str1 and str2 match at the current position.
            if str1[row] == str2[col]:
                # If the characters match, set the current cell value in matrix_row
                # to be 1 + up_left, simulating the top-left diagonal update.
                matrix_row[col] = 1 + up_left
                
                # Step 7: Update the max_value if the current match length is greater
                # than the previously found longest match.
                if matrix_row[col] > max_value:
                    max_value = matrix_row[col]
                    max_value_row = row  # Update the row where the longest match ends
            else:
                # If characters don't match, reset the current cell to 0
                # since there's no common substring ending at this position.
                matrix_row[col] = 0
                
            # Step 8: Update up_left to be the value of the current cell (saved earlier)
            # to be used in the next iteration for the next column.
            up_left = saved_current

    # Step 9: Calculate the start index of the longest common substring.
    # The longest substring starts at max_value_row - max_value + 1 and ends at max_value_row.
    # This is because max_value represents the length of the substring ending at max_value_row.
    start_index = max_value_row - max_value + 1

    # Step 10: Return the longest common substring by slicing str1
    # from the start_index up to (and including) max_value_row.
    return str1[start_index : max_value_row + 1]


# Main program: Solicit inputs from the user
if __name__ == "__main__":
    # Step 1: Input two strings from the user
    str1 = input("Enter the first string: ")
    str2 = input("Enter the second string: ")

    # Step 2: Call the Longest Common Substring function
    longest_substring = longest_common_substring_optimized(str1, str2)

    # Step 3: Print the result
    if longest_substring:
        print(f"The longest common substring is: '{longest_substring}'")
    else:
        print("No common substring found.")


# Explanation of the Code:
# ========================
#
# 1. Memory Optimization:
#    - This function is an optimized version of the longest common substring algorithm.
#    - Instead of using a full matrix of size `len(str1) * len(str2)`, we use a single row (`matrix_row`)
#      to store the lengths of common substrings for the current row.
#    - This reduces the space complexity from O(m * n) to O(n), where `n` is the length of `str2`.
#
# 2. Using `up_left` to Simulate Diagonal:
#    - The dynamic programming approach usually requires access to the upper-left diagonal cell (`matrix[i-1][j-1]`).
#    - To avoid storing the full matrix, the variable `up_left` is used to store the value of the previous row’s diagonal.
#    - Each time we loop through the inner columns (`col`), we store the current value in `matrix_row[col]` before
#      updating it. This value becomes the `up_left` for the next iteration.
#
# 3. Tracking the Longest Common Substring:
#    - Whenever a matching pair of characters is found (`str1[row] == str2[col]`), the corresponding
#      position in `matrix_row` is updated with `1 + up_left`, indicating the length of the common substring.
#    - The function then checks whether this newly calculated substring length (`matrix_row[col]`)
#      is greater than the previously stored `max_value`. If it is, the longest match so far is updated.
#
# 4. Extracting the Longest Substring:
#    - Once the matrix has been fully processed, the start index of the longest common substring is calculated
#      using the formula `start_index = max_value_row - max_value + 1`.
#    - The function then returns the longest common substring from `str1`, which is sliced from
#      `start_index` to `max_value_row + 1`.
#
# 5. Edge Cases:
#    - If no common substring exists, the function will return an empty string.
#
# Time Complexity:
# - The time complexity of this algorithm is O(m * n), where `m` is the length of `str1` and `n` is the length of `str2`.
# - Every character in `str1` is compared with every character in `str2`.
#
# Space Complexity:
# - The space complexity is O(n), where `n` is the length of `str2`.
# - Only one row of the matrix (`matrix_row`) is stored in memory, reducing memory usage significantly.
#
# Example Usage:
#
# Suppose we have the following strings:
# str1 = "abcde"
# str2 = "abfde"
#
# The function will calculate the longest common substring between them. 
# The longest common substring here is `"de"`, as it appears at the end of both strings.
#
# Output:
# The longest common substring is: "de"
