# V1.0 Minimum Viable Product

# def linear_search(target):
#   for product in products:
#       if product == target:
#           return True
#   return False

# products = ['apple', 'banana', 'carrot', 'date', 'eggplant', 'fig', 'grape', 'honeydew']
# target_product = input("Enter the name of the product to search: ")

# if linear_search(target_product) == True:
#   print('Product found!')
# else:
#   print('Product not found.')
#___________________________________________________________________________________________
#V 1.1 Hash Map Lookup
# def create_product_hashmap(products):
#   return {product: index for index, product in enumerate(products)}

# products = ['apple', 'banana', 'carrot', 'date', 'eggplant', 'fig', 'grape', 'honeydew']
# product_dict = create_product_hashmap(products)
# target_product = input("Enter the name of the product to search: ")

# if target_product in product_dict:
#   print(f'Product found at index {product_dict[target_product]}!')
# else:
#   print('Product not found.')
#___________________________________________________________________________________________

# V1.2 Case Non-Sensitivity
def create_product_hashmap(products):
  return {product.lower(): index for index, product in enumerate(products)}

products = ['apple', 'banana', 'carrot', 'date', 'eggplant', 'fig', 'grape', 'honeydew']
product_dict = create_product_hashmap(products)

target_product = input("Enter the name of the product to search: ").lower()

if target_product in product_dict:
  print(f'Product found at index {product_dict[target_product]}!')
else:
  print('Product not found.')
#___________________________________________________________________________________________