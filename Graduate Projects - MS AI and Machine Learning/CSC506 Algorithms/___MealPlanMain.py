from operator import attrgetter

# Constants
NUTRIENT_THRESHOLD = 0.001
FRACTION_THRESHOLD = 0.05
CALORIE_THRESHOLD = 0.1
MAX_CALORIES = 2000

# Food class to represent each food item
class Food:
    def __init__(self, name, protein, carbs, fat, calories):
        self.name = name
        self.protein = protein
        self.carbs = carbs
        self.fat = fat
        self.calories = calories
        self.fraction = 1.0

        # Calculate the calories for each macronutrient
        self.protein_calories = self.protein * 4
        self.carbs_calories = self.carbs * 4
        self.fat_calories = self.fat * 9

    # Method to set fraction of a food serving and recalculate calories
    def set_fraction(self, fraction):
        self.fraction = fraction
        self.protein_calories = self.protein * 4 * fraction
        self.carbs_calories = self.carbs * 4 * fraction
        self.fat_calories = self.fat * 9 * fraction
        self.calories = self.calories * fraction

    # String representation of a food object
    def __str__(self):
        return f"{self.fraction:.4f} serving of {self.name} (P={self.protein_calories:.1f}, C={self.carbs_calories:.1f}, F={self.fat_calories:.1f}, E={self.calories:.1f})"

# MealPlan class to represent a full meal plan
class MealPlan:
    def __init__(self):
        self.foods = []
        self.total_calories = 0
        self.total_protein_calories = 0
        self.total_carbs_calories = 0
        self.total_fat_calories = 0

    # Add food to the meal plan and update totals
    def add_food(self, food):
        self.foods.append(food)
        self.total_calories += food.calories
        self.total_protein_calories += food.protein_calories
        self.total_carbs_calories += food.carbs_calories
        self.total_fat_calories += food.fat_calories

    # Check what the total calories would be if the food were added
    def calories_with_food(self, food):
        return self.total_calories + food.calories

    # Get the current percentage of a nutrient (by calories)
    def percent_nutrient(self, nutrient):
        if self.total_calories == 0:
            return 0.0
        if nutrient == 'protein':
            return self.total_protein_calories / self.total_calories
        elif nutrient == 'carbs':
            return self.total_carbs_calories / self.total_calories
        elif nutrient == 'fat':
            return self.total_fat_calories / self.total_calories

    # Check if the meal plan meets the calorie limit
    def meets_calorie_limit(self, calorie_limit, threshold):
        return abs(self.total_calories - calorie_limit) <= threshold

    # Check if the meal plan meets the nutrient goal
    def meets_nutrient_goal(self, nutrient, goal, threshold):
        return abs(self.percent_nutrient(nutrient) - goal) <= threshold

    # Calculate fraction to fit calorie limit
    def fraction_to_fit_calorie_limit(self, food, calorie_limit):
        remaining_calories = calorie_limit - self.total_calories
        if remaining_calories >= food.calories:
            return 1.0
        return remaining_calories / food.calories

    # String representation of the meal plan
    def __str__(self):
        plan_str = "\nMeal Plan:\n"
        for food in self.foods:
            plan_str += f"{food}\n"
        plan_str += f"Total Calories: {self.total_calories:.1f}\n"
        plan_str += f"Protein: {self.percent_nutrient('protein'):.2f}\n"
        plan_str += f"Carbs: {self.percent_nutrient('carbs'):.2f}\n"
        plan_str += f"Fat: {self.percent_nutrient('fat'):.2f}\n"
        return plan_str

# Load food data from a file and create Food objects
def load_nutrient_data(filename):
    foods = []
    with open(filename, 'r') as file:
        for line in file:
            name, values = line.split(":")
            protein, carbs, fat, calories = map(float, values.split(","))
            food = Food(name.strip(), protein, carbs, fat, calories)
            foods.append(food)
    return foods

# Sort the food list based on the user's selected nutrient
def sort_food_list(foods, nutrient):
    if nutrient == 'protein':
        foods.sort(key=lambda food: food.protein_calories, reverse=True)
    elif nutrient == 'carbs':
        foods.sort(key=lambda food: food.carbs_calories, reverse=True)
    elif nutrient == 'fat':
        foods.sort(key=lambda food: food.fat_calories, reverse=True)

# Create a meal plan using a greedy algorithm to fit calorie and nutrient goals
def create_meal_plan(foods, nutrient, goal):
    plan = MealPlan()
    sort_food_list(foods, nutrient)

    for food in foods:
        if plan.calories_with_food(food) <= MAX_CALORIES:
            plan.add_food(food)
        else:
            fraction = plan.fraction_to_fit_calorie_limit(food, MAX_CALORIES)
            if fraction >= FRACTION_THRESHOLD:
                food.set_fraction(fraction)
                plan.add_food(food)

        if plan.meets_calorie_limit(MAX_CALORIES, CALORIE_THRESHOLD) and \
           plan.meets_nutrient_goal(nutrient, goal, NUTRIENT_THRESHOLD):
            break

    return plan

# Print the menu options
def print_menu():
    print()
    print("\t1 - Set maximum protein")
    print("\t2 - Set maximum carbohydrates")
    print("\t3 - Set maximum fat")
    print("\t4 - Exit program")
    print()

# Main program
def main():
    filename = input("Enter name of food data file: ")
    foods = load_nutrient_data(filename)

    while True:
        print_menu()
        choice = int(input("Enter choice (1-4): "))
        if choice == 4:
            break

        if choice == 1:
            nutrient = 'protein'
        elif choice == 2:
            nutrient = 'carbs'
        elif choice == 3:
            nutrient = 'fat'
        else:
            print("Invalid choice. Please try again.")
            continue

        goal = float(input(f"What percentage of calories from {nutrient} is the goal? ")) / 100.0

        plan = create_meal_plan(foods, nutrient, goal)
        print(plan)

if __name__ == "__main__":
    main()
