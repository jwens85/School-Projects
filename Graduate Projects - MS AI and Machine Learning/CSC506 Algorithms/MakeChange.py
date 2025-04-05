def make_change(amount):
  # Dictionary to store the count of each coin
  coins = {
      'quarters': 0,
      'dimes': 0,
      'nickels': 0,
      'pennies': 0
  }

  # Calculate the number of quarters
  while amount >= 25:
      coins['quarters'] += 1
      amount -= 25

  # Calculate the number of dimes
  while amount >= 10:
      coins['dimes'] += 1
      amount -= 10

  # Calculate the number of nickels
  while amount >= 5:
      coins['nickels'] += 1
      amount -= 5

  # Calculate the number of pennies
  while amount >= 1:
      coins['pennies'] += 1
      amount -= 1

  return coins

# Example usage:
amount = int(input("Enter the amount in cents: "))
change = make_change(amount)

print(f"Quarters: {change['quarters']}")
print(f"Dimes: {change['dimes']}")
print(f"Nickels: {change['nickels']}")
print(f"Pennies: {change['pennies']}")

"""
Explanation:
============

This program implements a greedy algorithm for making change using quarters, dimes, nickels, and pennies.
The algorithm aims to use the fewest number of coins possible by starting with the highest denomination (quarters)
and working its way down to the smallest (pennies).

### Components:
1. **Dictionary (coins):**
 - The `coins` dictionary is used to store the number of each type of coin (quarters, dimes, nickels, and pennies).

2. **Greedy Approach:**
 - The algorithm starts with the largest coin (quarter = 25 cents) and subtracts that value from the total amount
   as many times as possible.
 - Then, it proceeds to the next largest coin (dime = 10 cents) and repeats the process, followed by nickels and
   pennies.

3. **Subtraction Loops:**
 - Each loop checks if the remaining amount is greater than or equal to the coin's value (25, 10, 5, or 1 cent).
 - If the condition is true, it subtracts the coin's value from the amount and increments the count of that coin in the
   `coins` dictionary.

4. **Example Walkthrough:**
 - Let's say the user enters 87 cents:
   - The algorithm starts by subtracting 25 (quarters) three times (87 - 75 = 12 cents left).
   - It then subtracts 10 (dime) once (12 - 10 = 2 cents left).
   - Finally, it subtracts 2 pennies (2 - 2 = 0).
 - The result would be: 3 quarters, 1 dime, and 2 pennies.

### Output:
- After calculating the number of coins for each denomination, the result is printed out, showing how many quarters, dimes, nickels, and pennies are needed to make up the given amount.

### Limitations:
- The algorithm assumes that only quarters, dimes, nickels, and pennies are available. If there are other coin denominations
(such as 50-cent coins or dollar coins), the algorithm would need to be adjusted accordingly.
"""
