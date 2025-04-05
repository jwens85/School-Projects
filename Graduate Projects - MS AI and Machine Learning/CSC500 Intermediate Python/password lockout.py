correct_password = "secret"
login_attempts = 0
max_attempts = 3
while login_attempts < max_attempts:
    entered_password = input("Enter your password: ")
    if entered_password == correct_password:
        print("Login successful!")
        break
    login_attempts += 1
    print(f"Incorrect password. You have {max_attempts - login_attempts} attempts left.")
if login_attempts == max_attempts:
    print("Maximum login attempts exceeded. Account locked for security reasons.")
