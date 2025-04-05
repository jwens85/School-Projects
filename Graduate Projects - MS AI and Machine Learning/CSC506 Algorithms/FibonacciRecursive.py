def FibonacciNumber(termIndex):
  # Base case: If the index is 0, the Fibonacci number is 0
  if termIndex == 0:
      return 0
  # Base case: If the index is 1, the Fibonacci number is 1
  elif termIndex == 1:
      return 1
  # Recursive case: Sum of the two previous Fibonacci numbers
  else:
      return FibonacciNumber(termIndex - 1) + FibonacciNumber(termIndex - 2)

"""
Explanation:
============

This implementation of the Fibonacci sequence uses recursion to calculate the Fibonacci number at the specified index.

1. **Base Case**:
 - If the `termIndex` is `0`, the function returns `0`, as the 0th Fibonacci number is defined as 0.
 - If the `termIndex` is `1`, the function returns `1`, as the 1st Fibonacci number is defined as 1.

2. **Recursive Case**:
 - For any index greater than 1, the function calls itself twice:
   - `FibonacciNumber(termIndex - 1)`: Computes the (n-1)th Fibonacci number.
   - `FibonacciNumber(termIndex - 2)`: Computes the (n-2)th Fibonacci number.
 - The result is the sum of these two values, which gives the Fibonacci number for the given `termIndex`.

### Example Walkthrough:
If `termIndex = 5`, the function will perform the following calls:
- FibonacciNumber(5) calls:
- FibonacciNumber(4) + FibonacciNumber(3)
- FibonacciNumber(4) calls:
  - FibonacciNumber(3) + FibonacciNumber(2)
- And so on...

This results in calculating the Fibonacci number through multiple recursive calls until reaching the base cases (0 or 1).

### Efficiency:
- This recursive method has exponential time complexity O(2^n), which means it will become very slow for larger values of `termIndex`. 
- It recalculates Fibonacci numbers multiple times, which leads to inefficiency.

"""
