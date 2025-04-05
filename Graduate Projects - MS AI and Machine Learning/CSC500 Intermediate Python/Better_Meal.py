def meal():
    menu = {'chicken': 20, 'steak': 30, 'lobster': 40, 'cheesecake': 10, 'soda': 5, 'cocktail': 15}
    print("Menu Items:")
    for item, price in menu.items():
        print(f"{item.title()}: ${price}")
    order = input("What did you order? (use commas to separate multiple items): ").lower().split(', ')
    food_charge = sum(menu[item] for item in order if item in menu)
    service_level = input("How was the service? (Lousy, Fine, Great): ").lower()
    service_to_tip = {'lousy': 0.1, 'fine': 0.2, 'great': 0.3}
    tax_rate = 0.07
    tip_rate = service_to_tip.get(service_level, 0.18)
    tax_amount = food_charge * tax_rate
    tip_amount = food_charge * tip_rate
    total_amount = food_charge + tax_amount + tip_amount
    print(f"\nFood Cost: ${food_charge:.2f}")
    print(f"Tax Amount: ${tax_amount:.2f} (Tax Rate: {tax_rate * 100:.2f}%)")
    print(f"Tip Amount: ${tip_amount:.2f} (Tip Rate: {tip_rate * 100:.2f}%) based on '{service_level}' service")
    print(f"Total Cost of the Meal: ${total_amount:.2f}")
meal()