import math

# Point class to hold x, y coordinates
class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

# Function to calculate the Euclidean distance between two points
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Main program
# Read in x and y for Point P
p = Point()
p.x = int(input("Enter x coordinate of Point P: "))
p.y = int(input("Enter y coordinate of Point P: "))

# Read in num of steps to be taken in X and Y directions
x_step = int(input("Enter number of steps to move along X axis: "))
y_step = int(input("Enter number of steps to move along Y axis: "))

# Read in num of steps to be taken backwards every 3 steps
backward_step = int(input("Enter number of steps to move backwards every 3rd iteration: "))

# Output Point P
print(f"Point P: ({p.x},{p.y})")

# Start at the origin (0, 0)
current_point = Point(0, 0)
closest_point = Point(0, 0)
min_distance = distance(current_point, p)
iteration_count = 0
new_min_found = True

# Start moving step by step
while new_min_found:
    iteration_count += 1
    new_min_found = False  # Reset each loop

    # Move forward
    current_point.x += x_step
    current_point.y += y_step

    # Every 3rd iteration, move backward
    if iteration_count % 3 == 0:
        current_point.x -= backward_step
        current_point.y -= backward_step

    # Calculate the current distance to point P
    current_distance = distance(current_point, p)

    # Update the closest point if the current one is better
    if current_distance < min_distance:
        min_distance = current_distance
        closest_point = Point(current_point.x, current_point.y)
        new_min_found = True  # Keep looping if new minimum is found

# Output the final closest point, distance, and number of iterations
print(f"Arrival point: ({closest_point.x},{closest_point.y})")
print(f"Distance between P and arrival: {min_distance:.6f}")
print(f"Number of iterations: {iteration_count}")
