def room_1():
    while True:
        answer = input("Enter the correct number (1 to 5): ")
        if answer == "3":
            print("Correct! Moving to the next room...\n")
            break
        else:
            print("Incorrect, try again.")

def room_2():
    while True:
        answer = input("The more of this there is, the less you see. What is it?: ").lower()
        if answer == "darkness":
            print("Correct! Moving to the next room...\n")
            break
        else:
            print("Incorrect, try again.")

def room_3():
    while True:
        answer = input("Arrange the sequence: Sun, Earth, Moon. (Type as '1 2 3'): ")
        if answer == "2 1 3":
            print("Correct! You've escaped the mansion!\n")
            break
        else:
            print("Incorrect, try again.")

def start_game():
    print("Welcome to the Digital Escape Room! Solve the puzzles to escape.\n")
    room_1()
    room_2()
    room_3()
    print("Congratulations! You've successfully escaped the mansion.")

start_game()
