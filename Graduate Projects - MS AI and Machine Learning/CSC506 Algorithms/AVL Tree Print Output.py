class Node:
  def __init__(self, key):
      # Initialize a new node with the given key
      self.key = key
      self.left = None  # Left child of the node
      self.right = None  # Right child of the node
      self.height = 1  # Height of the node for balancing purposes

class AVLTree:
  def __init__(self):
      # Initialize an empty AVL tree
      self.root = None

  def insert(self, key):
      # Insert a new key into the AVL tree and return the new root
      if self.root is None:
          self.root = Node(key)
      else:
          self.root = self._insert_recursive(self.root, key)

  def _insert_recursive(self, current_node, key):
      # Recursive helper function for inserting a key
      if current_node is None:
          return Node(key)

      if key < current_node.key:
          current_node.left = self._insert_recursive(current_node.left, key)
      elif key > current_node.key:
          current_node.right = self._insert_recursive(current_node.right, key)
      else:
          return current_node  # Duplicate keys are not allowed in an AVL tree

      # Update the height of the current node
      current_node.height = 1 + max(self._get_height(current_node.left), self._get_height(current_node.right))

      # Get the balance factor and rebalance the tree if needed
      balance = self._get_balance(current_node)

      # Left Left Case
      if balance > 1 and key < current_node.left.key:
          return self._right_rotate(current_node)

      # Right Right Case
      if balance < -1 and key > current_node.right.key:
          return self._left_rotate(current_node)

      # Left Right Case
      if balance > 1 and key > current_node.left.key:
          current_node.left = self._left_rotate(current_node.left)
          return self._right_rotate(current_node)

      # Right Left Case
      if balance < -1 and key < current_node.right.key:
          current_node.right = self._right_rotate(current_node.right)
          return self._left_rotate(current_node)

      return current_node

  def remove(self, key):
      # Remove a node with a given key from the AVL tree
      if self.root is None:
          return None
      else:
          self.root = self._remove_recursive(self.root, key)

  def _remove_recursive(self, current_node, key):
      # Recursive helper function for removing a node
      if current_node is None:
          return current_node

      if key < current_node.key:
          current_node.left = self._remove_recursive(current_node.left, key)
      elif key > current_node.key:
          current_node.right = self._remove_recursive(current_node.right, key)
      else:
          # Node to be deleted found

          # Case 1: No children (leaf node)
          if current_node.left is None and current_node.right is None:
              return None

          # Case 2: One child
          elif current_node.left is None:
              return current_node.right
          elif current_node.right is None:
              return current_node.left

          # Case 3: Two children
          successor = self._find_min(current_node.right)
          current_node.key = successor.key
          current_node.right = self._remove_recursive(current_node.right, successor.key)

      # Update the height of the current node
      current_node.height = 1 + max(self._get_height(current_node.left), self._get_height(current_node.right))

      # Get the balance factor and rebalance the tree if needed
      balance = self._get_balance(current_node)

      # Left Left Case
      if balance > 1 and self._get_balance(current_node.left) >= 0:
          return self._right_rotate(current_node)

      # Left Right Case
      if balance > 1 and self._get_balance(current_node.left) < 0:
          current_node.left = self._left_rotate(current_node.left)
          return self._right_rotate(current_node)

      # Right Right Case
      if balance < -1 and self._get_balance(current_node.right) <= 0:
          return self._left_rotate(current_node)

      # Right Left Case
      if balance < -1 and self._get_balance(current_node.right) > 0:
          current_node.right = self._right_rotate(current_node.right)
          return self._left_rotate(current_node)

      return current_node

  def _left_rotate(self, z):
      # Perform a left rotation on the node z
      y = z.right
      T2 = y.left

      # Perform rotation
      y.left = z
      z.right = T2

      # Update heights
      z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
      y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

      return y

  def _right_rotate(self, z):
      # Perform a right rotation on the node z
      y = z.left
      T3 = y.right

      # Perform rotation
      y.right = z
      z.left = T3

      # Update heights
      z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
      y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

      return y

  def _get_height(self, node):
      # Return the height of the node
      if node is None:
          return 0
      return node.height

  def _get_balance(self, node):
      # Return the balance factor of the node
      if node is None:
          return 0
      return self._get_height(node.left) - self._get_height(node.right)

  def _find_min(self, current_node):
      # Find the minimum value node in the subtree
      while current_node.left is not None:
          current_node = current_node.left
      return current_node

  def inorder_traversal(self):
      # Perform in-order traversal of the tree and return the keys in sorted order
      result = []  # Initialize an empty list to store the result
      self._inorder_recursive(self.root, result)  # Call the recursive helper function
      return result  # Return the list of keys in sorted order

  def _inorder_recursive(self, current_node, result):
      # Recursive helper function for in-order traversal
      if current_node is not None:
          self._inorder_recursive(current_node.left, result)
          result.append(current_node.key)
          self._inorder_recursive(current_node.right, result)

  def print_tree(self):
      # Print the tree structure in a readable format
      self._print_recursive(self.root, 0)

  def _print_recursive(self, current_node, level):
      # Recursive function to print the tree structure
      if current_node is not None:
          self._print_recursive(current_node.right, level + 1)  # Print right subtree
          print('    ' * level + f'-> {current_node.key}')  # Print the current node with indentation
          self._print_recursive(current_node.left, level + 1)  # Print left subtree


# Soliciting input from the user
avl_tree = AVLTree()  # Create an instance of the AVL tree

# Ask the user to input values
print("Enter values to insert into the AVL Tree (comma-separated):")
values = input().split(',')  # Take comma-separated input from the user
values = [int(v.strip()) for v in values]  # Convert the input into a list of integers

# Insert the values into the AVL tree
for value in values:
  avl_tree.insert(value)

# Print the tree structure
print("\nAVL Tree structure:")
avl_tree.print_tree()

# Optionally, display the sorted order of the tree using in-order traversal
print("\nIn-order traversal (sorted order):")
print(avl_tree.inorder_traversal())
