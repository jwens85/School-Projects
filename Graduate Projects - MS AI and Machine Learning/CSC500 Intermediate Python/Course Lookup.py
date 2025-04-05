# Define dictionaries with key-value pairs
rooms = {
    "CSC101": "3004",
    "CSC102": "4501",
    "CSC103": "6755",
    "NET110": "1244",
    "COM241": "1411"
}
instructors = {
    "CSC101": "Haynes",
    "CSC102": "Alvarado",
    "CSC103": "Rich",
    "NET110": "Burke",
    "COM241": "Lee"
}
times = {
    "CSC101": "8:00 a.m.",
    "CSC102": "9:00 a.m.",
    "CSC103": "10:00 a.m.",
    "NET110": "11:00 a.m.",
    "COM241": "1:00 p.m."
}
# Lookup function
def query_course_database(course_number):
    try:
        room = rooms[course_number]
        instructor = instructors[course_number]
        time = times[course_number]
        return (True, room, instructor, time)  # Return a tuple with course info
    except KeyError:
        return (False, "Course not found.")  # Return an error message if the course is not found

# Main while loop that prompts user input and displays course info
while True:
    try:
        course_number = input("Enter a course number (e.g., CSC101), or 'quit' to exit: ")
        if course_number.lower() == 'quit':  # Allows the user to quit the loop
            break
        # Unpacks the returned tuple from the function into variables
        course_found, *info = query_course_database(course_number)
        if course_found:
            print(f"Room Number: {info[0]}")
            print(f"Instructor: {info[1]}")
            print(f"Meeting Time: {info[2]}")
        else:
            print(info[0])  #Prints an error message if the course is not found
    except Exception as e:
        print(f"An error occurred: {e}")#General exception handling
    finally:
        print("Good luck in your classes!")#Always executed, regardless of exceptions

print("Program ended. Have a great semester!")#Final message after breaking the loop
