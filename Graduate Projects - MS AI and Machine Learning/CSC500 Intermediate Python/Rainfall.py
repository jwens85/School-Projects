# Collect the number of years from the user
years = int(input("Enter the number of years: "))

# Initialize variables to hold total months and total rainfall
total_months = 0
total_rainfall = 0

# Outer loop for each year
for year in range(1, years + 1):
    # Inner loop for each month
    for month in range(1, 13):
# Collect rainfall data for the month
        rainfall = float(input(f"Enter the rainfall for year {year} month {month} in inches: "))
        total_rainfall += rainfall
        total_months += 1

# Calculate the average rainfall per month
average_rainfall = total_rainfall / total_months

# Display the results
print(f"Total number of months: {total_months}, Total inches of rainfall: {total_rainfall}, Average rainfall per month: {average_rainfall:.2f} inches")