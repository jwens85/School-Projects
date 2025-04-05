class User:
  def __init__(self, user_id, pin, balance, firstname, lastname):
      self.user_id = user_id
      self.pin = pin
      self.unsuccessful_attempts = 0
      self.is_locked = False
      self.balance = balance
      self.firstname = firstname
      self.lastname = lastname

  def authenticate(self, entered_pin):
      if self.is_locked:
          print("Account is locked.")
          return False

      if self.pin == entered_pin:
          self.reset_attempts()
          print("Authentication successful.")
          return True
      else:
          self.increment_attempts()
          print("Authentication failed.")
          if self.is_locked:
              print("Account has been locked due to too many unsuccessful attempts.")
          return False

  def increment_attempts(self):
      self.unsuccessful_attempts += 1
      if self.unsuccessful_attempts >= 3:
          self.lock_account()

  def reset_attempts(self):
      self.unsuccessful_attempts = 0

  def lock_account(self):
      self.is_locked = True

  def is_account_closed(self):
      return self.balance == 0

  def check_balance(self):
      if self.is_account_closed():
          print("Your account is closed due to a zero balance.")
          print("Please visit a branch to re-open your account.")
      else:
          print(f"Your balance is: ${self.balance}")

  def deposit(self, amount):
      if self.is_account_closed():
          print("Cannot deposit funds. Your account is closed due to a zero balance.")
          print("Please visit a branch to re-open your account.")
      elif amount > 0:
          self.balance += amount
          print(f"${amount} deposited successfully. New balance: ${self.balance}")
      else:
          print("Invalid deposit amount. Please enter a positive number.")

  def withdraw(self, amount):
      if amount > self.balance:
          print("Insufficient funds.")
      elif amount <= 0:
          print("Invalid withdrawal amount. Please enter a positive number.")
      elif amount == self.balance:
          print("Warning: Withdrawing all funds will close your account.")
          while True:
              confirmation = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
              if confirmation == 'yes':
                  self.balance -= amount
                  print(f"${amount} withdrawn successfully. Your account is now closed.")
                  break
              elif confirmation == 'no':
                  print("Withdrawal cancelled.")
                  break
              else:
                  print("Invalid input. Please enter 'yes' or 'no'.")
      else:
          self.balance -= amount
          print(f"${amount} withdrawn successfully. New balance: ${self.balance}")

def add_user(user_id, pin, balance, firstname, lastname):
  users[user_id] = User(user_id, pin, balance, firstname, lastname)

def authenticate_user(user_id, entered_pin):
  user = users.get(user_id)
  if user:
      return user.authenticate(entered_pin)
  else:
      print("User not found.")
      return False

def user_menu(user):
  while True:
      print("\nMenu:")
      print("1. Check Balance")
      print("2. Deposit Funds")
      print("3. Withdraw Funds")
      print("4. Exit")
      choice = input("Enter your choice: ")

      if choice == '1':
          user.check_balance()
      elif choice == '2':
          amount = float(input("Enter amount to deposit: "))
          user.deposit(amount)
      elif choice == '3':
          amount = float(input("Enter amount to withdraw: "))
          user.withdraw(amount)
      elif choice == '4':
          print("Thank you for using the ATM.")
          break
      else:
          print("Invalid choice. Please try again.")

users = {}

add_user("user1", "1234", 100, "John", "Wensink")
add_user("user2", "5678", 0, "Jane", "Smith")

def welcome_screen():
  while True:
      print("Welcome to the ATM!")
      user_id = input("Please enter your user ID: ")

      user = users.get(user_id)
      if not user:
          print("User not found.")
          continue

      if user.is_locked:
          print("Account is locked. Please visit a branch to unlock your account.")
          continue

      while not user.is_locked:
          entered_pin = input("Please enter your PIN: ")
          if user.authenticate(entered_pin):
              print("Welcome,", user.firstname, user.lastname)
              if user.is_account_closed():
                  print("Account is closed due to zero balance. Please visit a branch to re-open your account.")
                  break
              else:
                  user_menu(user)
                  break
          if user.is_locked:
              break

if __name__ == "__main__":
  welcome_screen()
