Python 3.11.2 (tags/v3.11.2:878ead1, Feb  7 2023, 16:38:35) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
 #        class ItemToPurchase:
 #          item_name=input()
 #          item_price=float(input())
 #          item_quantity=int(input())
 #          def __init__(self,item_name='none',item_price=0,item_quantity=0):
 #            self.item_name = item_name
 #            self.item_price = item_price
 #            self.item_quantity = item_quantity
 #          def print_item_cost(self):
 #            print(self.item_name, self.item_price, self.item_quantity)
 #        if __name__ == "__main__":
 #          print("Item 1")
 #          item1 = ItemToPurchase()
 #          print("Item 2")
 #          item2 = ItemToPurchase()
 #          print("TOTAL COST")
 #          item1.print_item_cost()
 #          item2.print_item_cost() 
 #       This doesn't work at all__________________________________________

 #       I'm going to try to start over, copy pasting the correct answer 
 # form the GPT,and removing items I don't understand to see how it breaks
 #       class ItemToPurchase:
 #         def __init__(self, item_name='none', item_price=0.0, item_quantity=0):
 #             self.item_name = item_name
 #             self.item_price = item_price
 #             self.item_quantity = item_quantity

 #         def print_item_cost(self):
 #             total_cost = self.item_price * self.item_quantity
 #             print(f"{self.item_name} {self.item_quantity} @ 
 # ${self.item_price} = ${total_cost}")
 #       if __name__ == "__main__":
 #          Prompt user for details of Item 1
 #         print("Item 1")
 #         item1_name = input("Enter the item name:\n")
 #         item1_price = float(input("Enter the item price:\n"))
 #         item1_quantity = int(input("Enter the item quantity:\n"))
 #          Create an object of ItemToPurchase
 #         item1 = ItemToPurchase(item1_name, item1_price, item1_quantity)

 #          Prompt user for details of Item 2
 #         print("\nItem 2")
 #         item2_name = input("Enter the item name:\n")
 #         item2_price = float(input("Enter the item price:\n"))
 #         item2_quantity = int(input("Enter the item quantity:\n"))
 #          Create another object of ItemToPurchase
 #         item2 = ItemToPurchase(item2_name, item2_price, item2_quantity)

 #          Calculate and print the total cost
 #         print("\nTOTAL COST")
 #         item1.print_item_cost()
 #         item2.print_item_cost()
 #         total_cost = item1.item_price * item1.item_quantity + item2.item_price * item2.item_quantity
 #         print(f"Total: ${total_cost}")
 #      _________________________________________________________________

 #      Let's try to hand type this from pseudo code without cheating

 #       class ItemToPurchase:
 #         def __init__ (self, item_name='none', item_price=0.0, item_quantity=0):
 #           self.item_name = item_name
 #           self.item_price = item_price
 #           self.item_quantity = item_quantity

 #       def print_item_cost():
 #         print(item_name, item_quantity, item_price, (item_quantity  * item_price))

 #       def main():
 #         items = []
 #         for i in range(2):
 #           print(i + 1)
 #           item_name = input("Eneter the item name:")
 #           item_price = float(input("Enter the item price:"))
 #           item_quantity = int(input("Enter the item quantity:"))
 #           items.append(ItemToPurchase(item_name, item_price, item_quantity))
 #           print("Total Cost: ")
 #           total_cost = 0
 #           for item in items:
 #             print(item_name, item_cost())
 #             total_cost = (item_price * item_quantity)
 #             print("Total: $", total_cost")
 #       main()
 #      _________________________________________________________________
 #      Not too bad, the predictive IDE still feels like cheating, but this is how it will be in the real world, it makes sense to use the tools that are available
 #      Let's debug this mess

 #      class ItemToPurchase:
 #        def __init__ (self, item_name='none', item_price=0.0, item_quantity=0):
 #          self.item_name = item_name
 #          self.item_price = item_price
 #          self.item_quantity = item_quantity

 #      def print_item_cost():
 #        print(item_name, item_quantity, item_price, (item_quantity  * item_price))

 #      def main():
 #        items = []
 #        for i in range(2):
 #          print(i + 1)
 #          item_name = input("Eneter the item name:")
 #          item_price = float(input("Enter the item price:"))
 #          item_quantity = int(input("Enter the item quantity:"))
 #          items.append(ItemToPurchase(item_name, item_price, item_quantity))
 #          print("Total Cost: ")
 #          total_cost = 0
 #          for item in items:
 #            print(item_name, item_price())
 #            total_cost = (item_price * item_quantity)
 #            print("Total: $", total_cost)
 #      main()
 #     ________________________________________________________________
 #      class ItemToPurchase:
 #        def __init__ (self, item_name='none', item_price=0.0, item_quantity=0):
 #          self.item_name = item_name
 #          self.item_price = item_price
 #          self.item_quantity = item_quantity

 #      def print_item_cost():
 #        print(item_name, item_quantity, item_price, (item_quantity  * item_price))

 #      def main():
 #        items = []
 #        for i in range(2):
 #          print(i + 1)
 #          item_name = input("Eneter the item name:")
 #          item_price = float(input("Enter the item price:"))
 #          item_quantity = int(input("Enter the item quantity:"))
 #          items.append(ItemToPurchase(item_name, item_price, item_quantity))
 #          print("Total Cost: ")
 #          total_cost = 0
 #          for item in items:
 #            print(item.item_name, item.item_price)
 #            total_cost = (item.item_price * item.item_quantity)
 #            print("Total: $", total_cost)
 #      print(item.item_name, item.item_quantity, item.item_price, (item.item_quantity  * item.item_price))

 #      main()
 #     ________________________________________________________________
 #      class ItemToPurchase:
 #          def __init__ (self, item_name='none', item_price=0.0, item_quantity=0):
 #              self.item_name = item_name
 #              self.item_price = item_price
 #              self.item_quantity = item_quantity

 #      def print_item_cost():
 #          print(item_name, item_quantity, item_price, (item_quantity * item_price))

 #      def main():
 #          items = []
 #          for i in range(2):
 #              print(i + 1)
 #              item_name = input("Enter the item name:")
 #              item_price = float(input("Enter the item price:"))
 #              item_quantity = int(input("Enter the item quantity:"))
 #              items.append(ItemToPurchase(item_name, item_price, item_quantity))
 #              print("Total Cost: ")
 #              total_cost = 0
 #              for item in items:
 #                  print(item.item_name, item.item_price)
 #                  total_cost = (item.item_price * item.item_quantity)
 #                  print("Total: $", total_cost)

 #      main()
 #     ________________________________________________________________
 #     Let's work on formatting now
 #     class ItemToPurchase:
 #         def __init__ (self, item_name='none', item_price=0.0, item_quantity=0):
 #             self.item_name = item_name
 #             self.item_price = item_price
 #             self.item_quantity = item_quantity

 #     def print_item_cost(item):
 #         print(item.item_name, item.item_quantity, item.item_price, (item.item_quantity * item.item_price))

 #     def print_total_cost(items):
 #         total_cost = 0
 #         for item in items:
 #             total_cost += item.item_quantity * item.item_price
 #         print("Total Cost:")
 #         print("Total: $", total_cost)

 #     def main():
 #         items = []
 #         for i in range(2):
 #             print(i + 1)
 #             item_name = input("Enter the item name:")
 #             item_price = float(input("Enter the item price:"))
 #             item_quantity = int(input("Enter the item quantity:"))
 #             items.append(ItemToPurchase(item_name, item_price, item_quantity))

 #         print("\nItems:")
 #         for item in items:
 #             print(item.item_name, item.item_price)

 #         for item in items:
 #             print_item_cost(item)

 #         print_total_cost(items)

 #     main()
 #     def print_item_cost(item):
 #         print(item.item_name, item.item_quantity, item.item_price, (item.item_quantity * item.item_price))
 #         print()

 #     def print_total_cost(items):
 #         total_cost = 0
 #         for item in items:
 #             total_cost += item.item_quantity * item.item_price
 #         print("Total Cost:")
 #         print("Total: $", total_cost)
 #         print()

 #     def main():
 #         items = []
 #         for i in range(2):
 #             print(i + 1)
 #             item_name = input("Enter the item name:")
 #             item_price = float(input("Enter the item price:"))
 #             item_quantity = int(input("Enter the item quantity:"))
 #             items.append(ItemToPurchase(item_name, item_price, item_quantity))

 #         print("\nItems:")
 #         for item in items:
 #             print(item.item_name, item.item_price)
 #         print()

 #         for item in items:
 #             print_item_cost(item)

 #         print_total_cost(items)

 #     main()
 #    ________________________________________________________________
 #    Looking better let's format the sections
 #     class ItemToPurchase:
 #         def __init__ (self, item_name='none', item_price=0.0, item_quantity=0):
 #             self.item_name = item_name
 #             self.item_price = item_price
 #             self.item_quantity = item_quantity

 #     def print_item_cost(item):
 #         print(item.item_name, item.item_quantity, item.item_price, (item.item_quantity * item.item_price))

 #     def print_total_cost(items):
 #         total_cost = 0
 #         for item in items:
 #             total_cost += item.item_quantity * item.item_price
 #         print("Total Cost:")
 #         print("Total: $", total_cost)

 #     def main():
 #         items = []
 #         for i in range(2):
 #             print(i + 1)
 #             item_name = input("Enter the item name:")
 #             item_price = float(input("Enter the item price:"))
 #             item_quantity = int(input("Enter the item quantity:"))
 #             items.append(ItemToPurchase(item_name, item_price, item_quantity))

 #         print("\nItems:")
 #         for item in items:
 #             print(item.item_name, item.item_price)

 #         for item in items:
 #             print_item_cost(item)

 #         print_total_cost(items)

 #     main()

 #    class ItemToPurchase:
 #        def __init__ (self, item_name='none', item_price=0.0, item_quantity=0):
 #            self.item_name = item_name
 #            self.item_price = item_price
 #            self.item_quantity = item_quantity

 #    def print_item_cost(item):
 #        print(item.item_name, item.item_quantity, item.item_price, (item.item_quantity * item.item_price))
 #        print()

 #    def print_total_cost(items):
 #        total_cost = 0
 #        for item in items:
 #            total_cost += item.item_quantity * item.item_price
 #        print("Total Cost:")
 #        print("Total: $", total_cost)
 #        return total_cost

 #    main()

 #    def main():
 #        items = []
 #        for i in range(2):
 #            print(i + 1)
 #            item_name = input("Enter the item name:")
 #            item_price = float(input("Enter the item price:"))
 #            item_quantity = int(input("Enter the item quantity:"))
 #            items.append(ItemToPurchase(item_name, item_price, item_quantity))

 #        print("\nItems:")
 #        for item in items:
 #            print(item.item_name, item.item_price)
 #        print()

 #        print("Item 1:")
 #        print_item_cost(items[0])

 #        print("Item 2:")
 #        print_item_cost(items[1])

 #        print("Totals:")
 #        print_total_cost(items)

 #    main()
 #    ________________________________________________________________
 #    Something broke let's try that again
 #   class ItemToPurchase:
 #       def __init__ (self, item_name='none', item_price=0.0, item_quantity=0):
 #           self.item_name = item_name
 #           self.item_price = item_price
 #           self.item_quantity = item_quantity
 #   def print_item_cost(item):
 #       print(item.item_name, item.item_quantity, item.item_price, (item.item_quantity * item.item_price))
 #       print()
 #   def print_total_cost(items):
 #       total_cost = 0
 #       for item in items:
 #           total_cost += item.item_quantity * item.item_price
 #       print("Total Cost:")
 #       print("Total: $", total_cost)
 #       return total_cost
 #   def main():
 #       items = []
 #       for i in range(2):
 #           print(i + 1)
 #           item_name = input("Enter the item name:")
 #           item_price = float(input("Enter the item price:"))
 #           item_quantity = int(input("Enter the item quantity:"))
 #           items.append(ItemToPurchase(item_name, item_price, item_quantity))

 #       print("\nItems:")
 #       for item in items:
 #           print(item.item_name, item.item_price)
 #       print()
 #       for item in items:
 #           print_item_cost(item)
 #       print_total_cost(items)
 #   main()
 #   ________________________________________________________________
 #   Let's format the floats for 2 decimals
 #   class ItemToPurchase:
 #       def __init__ (self, item_name='none', item_price=0.0, item_quantity=0):
 #           self.item_name = item_name
 #           self.item_price = "{:.2f}".format(item_price)
 #           self.item_quantity = item_quantity

 #   def print_item_cost(item):
 #       print(item.item_name, item.item_quantity, item.item_price, (item.item_quantity * float(item.item_price)))
 #       print()

 #   def print_total_cost(items):
 #       total_cost = 0
 #       for item in items:
 #           total_cost += item.item_quantity * float(item.item_price)
 #       print("Total Cost:")
 #       print("Total: $", "{:.2f}".format(total_cost))
 #       return total_cost

 #   def main():
 #       items = []
 #       for i in range(2):
 #           print(i + 1)
 #           item_name = input("Enter the item name:")
 #           item_price = float(input("Enter the item price:"))
 #           item_quantity = int(input("Enter the item quantity:"))
 #           items.append(ItemToPurchase(item_name, item_price, item_quantity))

 #       print("\nItems:")
 #       for item in items:
 #           print(item.item_name, item.item_price)
 #       print()

 #       for item in items:
 #           print_item_cost(item)

 #       print_total_cost(items)

 #   main()
 #  _________________________________________________________________
 #   I broke it again, let's try again

 #  class ItemToPurchase:
 #      def __init__(self, item_name='none', item_price=0.0, item_quantity=0):
 #          self.item_name = item_name
 #          self.item_price = "{:.2f}".format(item_price)
 #          self.item_quantity = item_quantity

 #  def print_item_cost(item):
 #      print(item.item_name, item.item_quantity, item.item_price, (item.item_quantity * float(item.item_price)))
 #      print()

 #  def print_total_cost(items):
 #      total_cost = 0
 #      for item in items:
 #          total_cost += item.item_quantity * float(item.item_price)
 #      print("Total Cost:")
 #      print("Total: $", "{:.2f}".format(total_cost))
 #      return total_cost

 #  def main():
 #      items = []
 #      for i in range(2):
 #          print(i + 1)
 #          item_name = input("Enter the item name:")
 #          item_price = float(input("Enter the item price:"))
 #          item_quantity = int(input("Enter the item quantity:"))
 #          items.append(ItemToPurchase(item_name, item_price, item_quantity))

 #      print("\nItems:")
 #      for item in items:
 #          print(item.item_name, item.item_price)
 #      print()

 #      for item in items:
 #          print_item_cost(item)

 #      print_total_cost(items)

 #  main()
 # _________________________________________________________________
 # Let's add some dollar signs
 # class ItemToPurchase:
 #     def __init__(self, item_name='none', item_price=0.0, item_quantity=0):
 #         self.item_name = item_name
 #         self.item_price = "${:.2f}".format(item_price)
 #         self.item_quantity = item_quantity

 # def print_item_cost(item):
 #     print(item.item_name, item.item_quantity, item.item_price, "${:.2f}".format(item.item_quantity * float(item.item_price)))
 #     print()

 # def print_total_cost(items):
 #     total_cost = 0
 #     for item in items:
 #         total_cost += item.item_quantity * float(item.item_price)
 #     print("Total Cost:")
 #     print("Total: $", "${:.2f}".format(total_cost))
 #     return total_cost

 # def main():
 #     items = []
 #     for i in range(2):
 #         print(i + 1)
 #         item_name = input("Enter the item name:")
 #         item_price = float(input("Enter the item price:"))
 #         item_quantity = int(input("Enter the item quantity:"))
 #         items.append(ItemToPurchase(item_name, item_price, item_quantity))

 #     print("\nItems:")
 #     for item in items:
 #         print(item.item_name, item.item_price)
 #     print()

 #     for item in items:
 #         print_item_cost(item)

 #     print_total_cost(items)

 # main()

# Now I'm getting weird ValueError: could not convert string to float issues 
#________________________________________________________________

class ItemToPurchase:
    def __init__(self, item_name='none', item_price=0.0, item_quantity=0):
        self.item_name = item_name
        self.item_price = item_price
        self.item_quantity = item_quantity

def print_item_cost(item):
    total_cost = item.item_quantity * item.item_price
    print(f"{item.item_name} {item.item_quantity} @ ${item.item_price:.2f} = ${total_cost:.2f}")

def print_total_cost(items):
...     total_cost = sum(item.item_quantity * item.item_price for item in items)
...     print("Total Cost:")
...     print(f"Total: ${total_cost:.2f}")
... 
... def main():
...     items = []
...     for i in range(2):
...         print(f"Item {i + 1}")
...         item_name = input("Enter the item name: ")
...         item_price = float(input("Enter the item price: "))
...         item_quantity = int(input("Enter the item quantity: "))
...         items.append(ItemToPurchase(item_name, item_price, item_quantity))
... 
...     print("\nTOTAL COST")
...     for item in items:
...         print_item_cost(item)
... 
...     print_total_cost(items)
... 
... main()
... 
... #This appears to be working now, and is sufficient for Week 4 Milestone
... 
... #A wishlist of things I would like to add for the Week 8 Final:
... 
... #I would like to see some error handling in the final project
... 
... #Perhaps work on multiple classes, having a shopping_cart class might make it possible to add/remove items before checkout
... 
... #A running total would be interesting and not too hard to implement
... 
... #Maybe overly ambitious, but ability to sync to an SQL database could be useful for real world applications
... 
... #More deatal on the receipt to look more like a real receipt, maybe discount codes or promotions could be fun
... 
... #Input validation would be necessary for security in the real world
