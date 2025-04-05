class Recipe:
  def __init__(self, name, ingredients):
      self.name = name  # Public attribute
      self.ingredients = ingredients  # Public attribute
      self.__secret_ingredient = "saffron"  # Name mangled private attribute

  def display_ingredients(self):
      print("The ingredients for", self.name, "are:", ', '.join(self.ingredients))

  def reveal_secret_ingredient(self):
      # This method is within the class, so it can access the name mangled attribute
      print("The secret ingredient is:", self.__secret_ingredient)

# Create an instance of the Recipe class
my_recipe = Recipe("Paella", ["rice", "chicken", "peas", "bell peppers"])

# Display the ingredients
my_recipe.display_ingredients()

# Attempt to reveal the secret ingredient using the method within the class
my_recipe.reveal_secret_ingredient()