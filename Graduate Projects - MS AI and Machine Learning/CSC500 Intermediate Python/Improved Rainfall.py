# Collect the number of years from the user, validate for an integer
while True:
    try:
        years = int(input("Enter the number of years: "))
        if years < 1:
            raise ValueError
        break
    except ValueError:
        print("Invalid input. Please enter a positive whole number.")
# Initialize variables and incorporate a flag for early exit
total_months = 0
total_rainfall = 0
exit_early = False
# Dictionary of month names for referencing in the prompt
month_names = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}
# Outer loop for each year
for year in range(1, years + 1):
    if exit_early:
        break
# Inner loop for each month using the month names dictionary
    for month in range(1, 13):  # 12 months in a year
# Collect rainfall data for the month, with input validation for a float
        while True:
            user_input = input(f"Rainfall for {month_names[month]} of Year {year} (inches): ")
            if user_input.upper() == "ESC":
                exit_early = True
                break
            try:
                rainfall = float(user_input)
                total_rainfall += rainfall
                total_months += 1
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        if exit_early:
            break
# Avoid division by zero if no months have been input
if total_months > 0:
# Calculate the average rainfall per month
    average_rainfall = total_rainfall / total_months

# Display the results on separate lines
    print("______________________________________")  
    print(f"Total months: {total_months}")
    print(f"Total rainfall: {total_rainfall} inches")
    print(f"Average monthly rainfall: {average_rainfall:.2f} inches")
else:
    print("No rainfall data was entered.")
