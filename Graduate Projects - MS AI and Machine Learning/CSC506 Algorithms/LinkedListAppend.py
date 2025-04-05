class Node:
  def __init__(self, data):
      self.data = data  # Store the value of the node
      self.next = None  # Reference to the next node (initially None)

class LinkedList:
  def __init__(self):
      self.head = None  # First node in the list (None if the list is empty)
      self.tail = None  # Last node in the list
      self.length = 0   # Tracks the number of nodes in the list

  def list_append(self, new_node):
      # If the list is empty, head and tail both point to the new node
      if self.head is None:
          self.head = new_node
          self.tail = new_node
      else:
          # Otherwise, link the current tail's 'next' to the new node
          self.tail.next = new_node
          # Update the tail to be the new node
          self.tail = new_node

      # Increase the list length
      self.length += 1

# Example usage:
# Create an empty list
linked_list = LinkedList()

# Create new nodes
node1 = Node(1)
node2 = Node(2)
node3 = Node(3)

# Append nodes to the list
linked_list.list_append(node1)
linked_list.list_append(node2)
linked_list.list_append(node3)

# Output the elements in the list
current = linked_list.head
while current:
  print(current.data)  # Output will be 1, 2, 3
  current = current.next
