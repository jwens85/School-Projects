class Node:
  def __init__(self, data):
      self.data = data
      self.next = None

class SinglyLinkedList:
  def __init__(self):
      self.head = None  # Starting with the head of the list being None

  def append(self, data):
      """Append a node with the provided data to the end of the list."""
      new_node = Node(data)
      if self.head is None:
          self.head = new_node
          return
      last_node = self.head
      while last_node.next:
          last_node = last_node.next
      last_node.next = new_node

  def print_list(self):
      """Print all elements in the list."""
      current = self.head
      while current:
          print(current.data)
          current = current.next

  def is_empty(self):
      """Check if the list is empty."""
      return self.head is None

  def insert_at_beginning(self, data):
      """Insert a new node at the beginning of the list."""
      new_node = Node(data)
      new_node.next = self.head
      self.head = new_node

# Usage:
sll = SinglyLinkedList()
sll.append('eggs')
sll.append('ham')
sll.append('spam')
sll.print_list()
