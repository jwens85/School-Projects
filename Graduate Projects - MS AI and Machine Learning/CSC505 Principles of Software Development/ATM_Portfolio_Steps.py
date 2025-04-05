def authenticate_user_steps():
  print("1. User inserts ATM card.")
  print("2. ATM prompts user to enter PIN.")
  print("3. User enters PIN.")
  print("4. ATM verifies PIN.")
  print("5. If PIN is correct, authentication is successful.")
  print("6. If PIN is incorrect, attempt count increases.")
  print("7. After 3 incorrect attempts, account is locked.")
  print("8. If account is locked, user is notified, and authentication fails.")

def check_balance_steps():
  print("1. User selects 'Check Balance' option from the ATM menu.")
  print("2. ATM retrieves the user's balance from their account.")
  print("3. ATM displays the current balance on the screen.")
  print("4. User can choose to perform another operation or exit.")

def deposit_funds_steps():
  print("1. User selects 'Deposit Funds' option from the ATM menu.")
  print("2. ATM prompts the user to enter the amount to deposit.")
  print("3. User enters the amount.")
  print("4. ATM asks the user to confirm the deposit amount.")
  print("5. If confirmed, ATM adds the amount to the user's account balance.")
  print("6. ATM prints a receipt showing the new balance.")
  print("7. User can choose to perform another operation or exit.")

def withdraw_funds_steps():
  print("1. User selects 'Withdraw Funds' option from the ATM menu.")
  print("2. ATM prompts the user to enter the amount to withdraw.")
  print("3. User enters the amount.")
  print("4. ATM checks if the user has sufficient funds.")
  print("5. If sufficient funds, ATM dispenses the cash.")
  print("6. ATM deducts the withdrawn amount from the user's account.")
  print("7. ATM prints a receipt showing the new balance.")
  print("8. If the user withdraws all funds, the account is closed.")
  print("9. User can choose to perform another operation or exit.")

def lock_account_steps():
  print("1. If a user fails to authenticate after 3 attempts, the account is locked.")
  print("2. ATM notifies the user that their account is locked.")
  print("3. User is instructed to visit a bank branch to unlock their account.")
  print("4. ATM exits the authentication process.")

def main():
  print("ATM Operations Steps:")
  print("----------------------")
  authenticate_user_steps()
  print("\n")
  check_balance_steps()
  print("\n")
  deposit_funds_steps()
  print("\n")
  withdraw_funds_steps()
  print("\n")
  lock_account_steps()
  print("----------------------")
  print("End of ATM Operations Steps.")

if __name__ == "__main__":
  main()
