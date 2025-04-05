class Node:
  def __init__(self, data=None):
      self.data = data
      self.next = None
      self.prev = None

class DoublyLinkedList:
  def __init__(self):
      self.head = None
      self.tail = None
      self.count = 0

  def append(self, data):
      """Append an item to the end of the list."""
      new_node = Node(data)
      if not self.head:  # If the list is empty
          self.head = self.tail = new_node
      else:
          new_node.prev = self.tail
          self.tail.next = new_node
          self.tail = new_node
      self.count += 1

  def prepend(self, data):
      """Prepend an item to the beginning of the list."""
      new_node = Node(data)
      if not self.head:  # If the list is empty
          self.head = self.tail = new_node
      else:
          new_node.next = self.head
          self.head.prev = new_node
          self.head = new_node
      self.count += 1

  def delete(self, data):
      """Delete a node from the list containing the given data."""
      current = self.head
      while current:
          if current.data == data:
              if current.prev:
                  current.prev.next = current.next
              if current.next:
                  current.next.prev = current.prev
              if current == self.head:  # Move head if needed
                  self.head = current.next
              if current == self.tail:  # Move tail if needed
                  self.tail = current.prev
              self.count -= 1
              return True
          current = current.next
      return False  # Data not found

  def display(self):
      """Display the list."""
      elements = []
      current = self.head
      while current:
          elements.append(current.data)
          current = current.next
      print("List: ", elements)

# Example Usage
dll = DoublyLinkedList()
dll.append('eggs')
dll.append('ham')
dll.prepend('spam')
dll.display()
dll.delete('ham')
dll.display()
