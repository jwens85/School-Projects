class ListyApp:
  def __init__(self):
      self.pages = []

  def add_page(self, name, next_page=None):
      self.pages.append({"name": name, "next": next_page})

  def remove_page(self, name):
      self.pages = [page for page in self.pages if page['name'] != name]

  def print_details(self):
      print("Listy App Prototype Details")
      print("Total number of pages:", len(self.pages))
      print("\nPage Names and Flow Sequence:")
      for page in self.pages:
          print(f"Page Name: {page['name']}")
          if page['next']:
              print(f"   Flows to: {page['next']}")
          else:
              print("   This is the last page in the flow.")

if __name__ == "__main__":
  app = ListyApp()
  app.add_page("Sign In", "Home Screen")
  app.add_page("Home Screen", "List Screen")
  app.add_page("List Screen", "Edit Item Screen")
  app.add_page("Edit Item Screen", "Share Screen")
  app.add_page("Share Screen")
  app.print_details()