def LongestCommonSubstringOptimized(str1, str2, required_match_length=4):
    # Step 1: Initialize variables to store the previous row and track the longest substring
    # prev_row keeps track of the last row of the matrix (to avoid storing the entire matrix)
    # longest_length keeps track of the length of the longest valid substring found
    # end_row stores the row index where the longest valid substring ends in str1
    prev_row = [0] * len(str2)
    longest_length = 0
    end_row = 0

    # Step 2: Iterate through each character in str1 and str2
    for row in range(len(str1)):
        # Initialize a new row (current_row) for the current character in str1
        current_row = [0] * len(str2)
        for col in range(len(str2)):
            # Step 3: If the characters match, compute the value for the current cell
            if str1[row] == str2[col]:
                # If we're not at the first row or first column,
                # take the value from the previous row (up-left diagonal cell) and add 1.
                if row > 0 and col > 0:
                    current_row[col] = prev_row[col - 1] + 1
                else:
                    current_row[col] = 1

                # Step 4: Update the length of the longest substring found
                # Ensure the match is at least as long as the required_match_length
                if current_row[col] >= required_match_length and current_row[col] > longest_length:
                    longest_length = current_row[col]  # Update longest length
                    end_row = row  # Store the row index where the longest substring ends
            else:
                # No common substring at this position, so set the value to 0
                current_row[col] = 0

        # Step 5: Update the previous row to be the current row for the next iteration
        prev_row = current_row

    # Step 6: Calculate the start index of the longest common substring in str1
    # The start index is calculated by subtracting the length of the substring from the end_row.
    start_index = end_row - longest_length + 1

    # Step 7: Return the longest common substring from str1, if it's longer than or equal to the required length
    if longest_length >= required_match_length:
        return str1[start_index:start_index + longest_length]
    else:
        return ""  # Return an empty string if no valid substring is found


# Main program to take inputs from the user
if __name__ == "__main__":
    # Step 1: Input two DNA sequences from the user
    str1 = input("Enter the first DNA sequence: ")
    str2 = input("Enter the second DNA sequence: ")

    # Step 2: Call the LongestCommonSubstringOptimized function with a required match length of 4
    longest_substring = LongestCommonSubstringOptimized(str1, str2, required_match_length=4)

    # Step 3: Print the result
    if longest_substring:
        print(f"The longest common substring is: '{longest_substring}'")
    else:
        print("No common substring of 4 or more characters found.")

"""
### How the Program Works:
==========================

This program finds the **longest common substring** between two DNA sequences provided by the user, but only counts matches if the substring is at least 4 characters long.

1. **Matrix Construction**:
   - A dynamic programming matrix is simulated using two rows (`prev_row` and `current_row`) to reduce memory usage. The matrix is used to store the length of the longest common substring that ends at each position in `str1` and `str2`.

2. **Character Matching**:
   - We compare characters from both strings, `str1` and `str2`, one by one.
   - If the characters match, the length of the common substring is increased by 1, based on the value from the "diagonal" position (up-left) in the previous row.
   - If the characters don’t match, the length of the common substring at that position is reset to 0.

3. **Tracking the Longest Valid Substring**:
   - While processing the matrix, we keep track of the **longest valid substring** found so far. To be considered valid, the substring must be at least 4 characters long (this threshold can be adjusted using the `required_match_length` parameter).
   - The program stores the length and the position where the longest substring ends.

4. **Extracting the Longest Substring**:
   - Once the matrix processing is complete, the program extracts the longest common substring from `str1` using the stored position and length of the match.

5. **Memory Optimization**:
   - Instead of storing the entire matrix, only two rows are stored in memory at any time: the current row and the previous row.
   - This reduces the memory complexity to **O(n)**, where `n` is the length of the second string (`str2`), making the program more efficient for large sequences.

### Example:
==============
For example, if the user enters the following DNA sequences:

str1 = "ATCG GACA TCAG GACG CGTA"
str2 = "ATAC GACA TCAG GAGC CGTA"

The longest common substring of 4 or more characters would be:
'GACA TCAG'

### Time Complexity:
====================
The program runs with **O(m * n)** time complexity, where:
- `m` is the length of `str1`
- `n` is the length of `str2`

### Space Complexity:
=====================
The space complexity is **O(n)**, where `n` is the length of the second string `str2`. This is because we only store two rows of the dynamic programming matrix at a time.

"""

