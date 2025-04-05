def create_product_hashmap(products):
  return {product.lower(): index for index, product in enumerate(products)}

products = ['apple', 'banana', 'carrot', 'date', 'eggplant', 'fig', 'grape', 'honeydew']
product_dict = create_product_hashmap(products)

target_product = input("Enter the name of the product to search: ").lower()

if target_product in product_dict:
  print(f'Product found at index {product_dict[target_product]}!')
else:
  print('Product not found.')