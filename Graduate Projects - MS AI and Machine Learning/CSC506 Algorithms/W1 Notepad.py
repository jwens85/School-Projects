#V1.1 Add a Hash Index to the Function
def create_product_index(products):
  product_index = {}
  for index, product in enumerate(products):
      product_index[product] = index
  return product_index

def indexed_search(product_index, target):
  if target in product_index:
      return product_index[target]
  else:
      return -1

products = ['apple', 'banana', 'carrot', 'date', 'eggplant', 'fig', 'grape', 'honeydew']

product_index = create_product_index(products)

target_product = input("Enter the name of the product to search: ")

result = indexed_search(product_index, target_product)

if result != -1:
  print(f'Product found at index {result}')
else:
  print('Product not found.')
