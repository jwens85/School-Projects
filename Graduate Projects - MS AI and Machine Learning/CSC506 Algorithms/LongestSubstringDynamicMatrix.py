def LongestCommonSubstring(str1, str2):
  # Step 1: Initialize a matrix to store lengths of longest common suffixes
  # between substrings of str1 and str2.
  # matrix[i][j] will contain the length of the longest common substring
  # that ends at str1[i-1] and str2[j-1].

  matrix = [[0] * len(str2) for _ in range(len(str1))]

  # Variable to track the length of the longest common substring
  longest_length = 0

  # Variable to store the end position (row) of the longest substring in str1
  end_row = 0

  # Step 2: Iterate through each character in str1 and str2
  for row in range(len(str1)):
      for col in range(len(str2)):
          # Step 3: If the characters match, compute the value for the current cell
          if str1[row] == str2[col]:
              # If we're not at the first row and first column,
              # take the value from the cell up-left and add 1.
              # Otherwise, the value is just 1.
              if row > 0 and col > 0:
                  matrix[row][col] = matrix[row - 1][col - 1] + 1
              else:
                  matrix[row][col] = 1

              # Step 4: Update the length of the longest substring found
              if matrix[row][col] > longest_length:
                  longest_length = matrix[row][col]
                  end_row = row
          else:
              # If characters don't match, this substring is not common
              # so the value is 0.
              matrix[row][col] = 0

  # Step 5: Calculate the start index of the longest common substring
  start_index = end_row - longest_length + 1

  # Step 6: Return the longest common substring from str1
  return str1[start_index:start_index + longest_length]

"""
Explanation:
============

This algorithm finds the **longest common substring** between two strings `str1` and `str2`. 
A dynamic programming matrix is used to store the lengths of the longest common substrings 
that end at specific positions in the two strings. The result is built by comparing characters 
of the two strings and keeping track of matching substrings.

### Step-by-Step Breakdown:

1. **Matrix Initialization**:
 - We initialize a 2D matrix (`matrix`) of size `len(str1)` x `len(str2)`, where each element starts at 0.
 - This matrix will store the lengths of the longest common suffixes for substrings ending at each position.

2. **Iterating Through Characters**:
 - We loop through each character in `str1` and `str2`.
 - If the characters match (`str1[row] == str2[col]`), the value at `matrix[row][col]` is the value of the cell diagonally up-left (`matrix[row-1][col-1]`) plus 1.
 - If we are in the first row or first column, the value is set to 1 directly.

3. **Tracking the Longest Substring**:
 - While filling in the matrix, we keep track of the longest substring found by storing the maximum value and its position in the matrix.

4. **Determining the Start of the Longest Substring**:
 - The longest common substring ends at the position stored in `end_row`.
 - The start of the longest common substring is determined by subtracting the length of the substring from `end_row`.

5. **Extracting the Longest Substring**:
 - The substring is extracted from `str1` starting at `start_index` and having a length of `longest_length`.

### Example:
Given two strings:
- `str1 = "abcde"`
- `str2 = "abfde"`

The matrix would look like this (matching characters are marked):
