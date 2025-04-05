# Collect the number of books purchased from the user
books_purchased = int(input("Enter the number of books purchased this month: "))

# Initialize points variable
points = 0

# Determine the points awarded based on the number of books
if books_purchased < 2:
    points = 0
elif 2 <= books_purchased < 4:
    points = 5
elif 4 <= books_purchased < 6:
    points = 15
elif 6 <= books_purchased < 8:
    points = 30
else:
    points = 60  # For 8 or more books

# Display the points awarded
print(f"Points awarded: {points}")
