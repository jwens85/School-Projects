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

        if self.is_account_closed():
            print("Account is closed due to zero balance.")
            return False

        if self.pin == entered_pin:
            self.reset_attempts()
            print("Authentication successful.")
            return True
        else:
            self.increment_attempts()
            print("Authentication failed.")
            return False

    def increment_attempts(self):
        self.unsuccessful_attempts += 1
        if self.unsuccessful_attempts >= 3:
            self.lock_account()

    def reset_attempts(self):
        self.unsuccessful_attempts = 0

    def lock_account(self):
        self.is_locked = True
        print("Account has been locked due to too many unsuccessful attempts.")

    def is_account_closed(self):
        return self.balance == 0


def add_user(user_id, pin, balance, firstname, lastname):
    users[user_id] = User(user_id, pin, balance, firstname, lastname)

def authenticate_user(user_id, entered_pin):
    user = users.get(user_id)
    if user:
        return user.authenticate(entered_pin)
    else:
        print("User not found.")
        return False

users = {}

add_user("user1", "1234", 100, "John", "Wensink")
add_user("user2", "5678", 0, "Jane", "Smith")

def welcome_screen():
    print("Welcome to the ATM!")
    user_id = input("Please enter your user ID: ")

    user = users.get(user_id)
    if not user:
        print("User not found.")
        return

    while not user.is_locked:
        entered_pin = input("Please enter your PIN: ")
        if user.authenticate(entered_pin):
            print("Welcome,", user.firstname, user.lastname)
            break
        elif user.is_locked:
            print("Account has been locked due to too many unsuccessful attempts.")
            break

if __name__ == "__main__":
    welcome_screen()