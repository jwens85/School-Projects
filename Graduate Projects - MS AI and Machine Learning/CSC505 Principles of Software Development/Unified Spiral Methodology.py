class Wensink:
  def __init__(self):
         self.elements = {
          'Communication': [],
          'Planning': [],
          'Modeling': [],
          'Construction': [],
          'Deployment': []
      }
         self.updates = {element: {} for element in self.elements}

  def input_element_descriptions(self):
      """Prompts user to input descriptions for each predefined diagram element."""
      print("Please enter descriptions for the following diagram elements. Type 'done' when finished with each element.")
      for element in self.elements:
          print(f"Enter descriptions for {element}:")
          while True:
              description = input("Description (type 'done' to finish): ")
              if description.lower() == 'done':
                  break
              self.elements[element].append(description)
              self.updates[element][description] = []

  def collect_updates(self):
            for i in range(1, 6):
          print(f"\n--- Loop {i}: Enter updates for each description ---")
          for element, descriptions in self.elements.items():
              for description in descriptions:
                  update = input(f"Element: {element}\nDescription: {description}\nUpdate: ")
                  self.updates[element][description].append(update)

  def display_elements(self):
      """Displays the elements, their descriptions, and updates in a formatted output."""
      print("\nFormatted Diagram Elements and Updates:")
      for element, descriptions in self.elements.items():
          print(f"\n- {element}:")
          for description in descriptions:
              print(f"  * {description}")
              for idx, update in enumerate(self.updates[element][description], start=1):
                  print(f"    - Loop {idx}: {update}")

if __name__ == "__main__":
  wensink_diagram = Wensink()
  wensink_diagram.input_element_descriptions()
  wensink_diagram.collect_updates()
  wensink_diagram.display_elements()
