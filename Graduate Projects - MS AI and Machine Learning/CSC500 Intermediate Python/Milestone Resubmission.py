class ItemToPurchase:
# Initialize an item that has a name, price, and quantity
  def __init__(self, item_name='none', item_price=0.0, item_quantity=0):
      self.item_name = item_name
      self.item_price = item_price
      self.item_quantity = item_quantity

# Function to print the cost of an item with a formatted string literal
def print_item_cost(item):
  total_cost = item.item_quantity * item.item_price  
  print(f"{item.item_name} {item.item_quantity} @ ${item.item_price:.2f} "
        f"= ${total_cost:.2f}")

# Calculates and prints the total cost of all items
def print_total_cost(items):
  print("TOTAL COST")
  total_cost = 0  # Initialize total cost
# Loop to print the cost of an item and iterate to add to the total cost
  for item in items:
      print_item_cost(item)
      total_cost += item.item_quantity * item.item_price
  print(f"Total: ${total_cost:.2f}")

# Define the main function creating a list and starting an item counter
def main():
  items = []
  item_count = 0

  while item_count < 2:
      print(f"\nItem {item_count + 1}")
      item_name = input("Enter the item name: ")

# Validate item price input
      while True:
          item_price = input("Enter the item price: ")
          try:
              item_price = float(item_price)
              if item_price < 0:
                  raise ValueError
              break
          except ValueError:
              print("Invalid input. Please enter a positive number for the price.")

# Validate item quantity input
      while True:
          item_quantity = input("Enter the item quantity: ")
          try:
              item_quantity = int(item_quantity)
              if item_quantity < 0:
                  raise ValueError
              break
          except ValueError:
              print("Invalid input. Please enter a positive number for the quantity.")

      items.append(ItemToPurchase(item_name, item_price, item_quantity))
      item_count += 1

  print_total_cost(items)

# Run the program
main()
