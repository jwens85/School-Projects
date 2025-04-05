import numpy as np

shopping_cart = np.array([], dtype=object)

def add_to_cart(product, price):
    global shopping_cart
    item = {'product': product, 'price': price}  
    shopping_cart = np.append(shopping_cart, item)  

def show_cart():
    if shopping_cart.size == 0:  # 

        print("Your shopping cart is empty.")
    else:
        print("Items in your shopping cart:")
        for item in shopping_cart:  
            print(f"Product: {item['product']}, Price: ${item['price']}")

add_to_cart('Steak', 21.00)
add_to_cart('Asparagus', 6.00)
add_to_cart('Potatoes', 3.00)

show_cart()
