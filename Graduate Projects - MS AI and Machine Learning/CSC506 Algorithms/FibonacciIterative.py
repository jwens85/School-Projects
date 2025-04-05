def FibonacciNumber(termIndex):
  # Base case: If the index is 0, the Fibonacci number is 0
  if termIndex == 0:
      return 0

  # Initializing the first two Fibonacci numbers
  previous = 0  # This is Fib(0)
  current = 1   # This is Fib(1)
  i = 1

  # Iteratively calculate the next Fibonacci numbers up to the required index
  while i < termIndex:
      next = previous + current  # Calculate next Fibonacci number
      previous = current         # Move current to previous for the next iteration
      current = next             # Update current to the new next value
      i += 1                     # Move to the next index

  # Return the Fibonacci number at the given index
  return current

"""
Explanation:
============

This implementation of the Fibonacci sequence uses **iteration** (a dynamic programming approach) to calculate the Fibonacci number at the specified index.

1. **Base Case**:
 - If the `termIndex` is `0`, the function immediately returns `0` because the 0th Fibonacci number is defined as 0.

2. **Iterative Calculation**:
 - The first two Fibonacci numbers, `previous` and `current`, are initialized to `0` and `1`, respectively.
   - `previous` holds Fib(0), and `current` holds Fib(1).
 - The loop runs from index `1` up to `termIndex - 1`:
   - At each iteration:
     - The next Fibonacci number is calculated as the sum of `previous` and `current`.
     - `previous` is updated to the current value (shifting one step back in the sequence).
     - `current` is updated to the newly calculated value.
   - The loop iterates until `current` holds the Fibonacci number at the `termIndex`.

3. **Returning the Result**:
 - Once the loop finishes, `current` holds the Fibonacci number for the desired `termIndex`, which is then returned.

### Example Walkthrough:
For `termIndex = 5`:
- First iteration:
- `previous = 0`, `current = 1`, next = 0 + 1 = 1
- Second iteration:
- `previous = 1`, `current = 1`, next = 1 + 1 = 2
- Third iteration:
- `previous = 1`, `current = 2`, next = 1 + 2 = 3
- Fourth iteration:
- `previous = 2`, `current = 3`, next = 2 + 3 = 5
- Returns 5 for `termIndex = 5`.

### Efficiency:
- This iterative method has a linear time complexity O(n), which makes it **much faster** than the recursive version for large `termIndex` values.
- It only requires constant space (O(1)) for `previous` and `current`, making it very memory-efficient.

"""
