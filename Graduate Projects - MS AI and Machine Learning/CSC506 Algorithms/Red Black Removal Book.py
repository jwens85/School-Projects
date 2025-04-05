def _bst_remove(self, key):
  node = self.search(key)
  self._bst_remove_node(node)

def _bst_remove_node(self, node):
  if node is None:
      return

  # Case 1: Internal node with 2 children
  if node.left is not None and node.right is not None:
      # Find successor
      successor_node = node.right
      while successor_node.left is not None:
          successor_node = successor_node.left

      # Copy successor's key
      successor_key = successor_node.key

      # Recursively remove successor
      self._bst_remove_node(successor_node)

      # Set node's key to copied successor key
      node.key = successor_key

  # Case 2: Root node (with 1 or 0 children)
  elif node is self.root:
      if node.left is not None:
          self.root = node.left
      else:
          self.root = node.right

      # Make sure the new root, if not None, has parent set to None
      if self.root is not None:
          self.root.parent = None

  # Case 3: Internal with left child only
  elif node.left is not None:
      node.parent.replace_child(node, node.left)

  # Case 4: Internal with right child OR leaf
  else:
      node.parent.replace_child(node, node.right)

def is_none_or_black(self, node):
  if node is None:
      return True
  return node.is_black()

def is_not_none_and_red(self, node):
  if node is None:
      return False
  return node.is_red()

def prepare_for_removal(self, node):
  if self.try_case1(node):
      return

  sibling = node.get_sibling()
  if self.try_case2(node, sibling):
      sibling = node.get_sibling()
  if self.try_case3(node, sibling):
      return
  if self.try_case4(node, sibling):
      return
  if self.try_case5(node, sibling):
      sibling = node.get_sibling()
  if self.try_case6(node, sibling):
      sibling = node.get_sibling()

  sibling.color = node.parent.color
  node.parent.color = "black"
  if node is node.parent.left:
      sibling.right.color = "black"
      self.rotate_left(node.parent)
  else:
      sibling.left.color = "black"
      self.rotate_right(node.parent)

def remove(self, key):
  node = self.search(key)
  if node is not None:
      self.remove_node(node)
      return True
  return False

def remove_node(self, node):
  if node.left is not None and node.right is not None:
      predecessor_node = node.get_predecessor()
      predecessor_key = predecessor_node.key
      self.remove_node(predecessor_node)
      node.key = predecessor_key
      return

  if node.is_black():
      self.prepare_for_removal(node)
  self._bst_remove(node.key)

  # One special case if the root was changed to red
  if self.root is not None and self.root.is_red():
      self.root.color = "black"

def search(self, key):
  current_node = self.root
  while current_node is not None:
      # Return the node if the key matches.
      if current_node.key == key:
          return current_node

      # Navigate to the left if the search key is
      # less than the node's key.
      elif key < current_node.key:
          current_node = current_node.left

      # Navigate to the right if the search key is
      # greater than the node's key.
      else:
          current_node = current_node.right

  # The key was not found in the tree.
  return None

def try_case1(self, node):
  if node.is_red() or node.parent is None:
      return True
  return False # node case 1

def try_case2(self, node, sibling):
  if sibling.is_red():
      node.parent.color = "red"
      sibling.color = "black"
      if node is node.parent.left:
          self.rotate_left(node.parent)
      else:
          self.rotate_right(node.parent)
      return True
  return False # not case 2

def try_case3(self, node, sibling):
  if node.parent.is_black() and sibling.are_both_children_black():
      sibling.color = "red"
      self.prepare_for_removal(node.parent)
      return True
  return False # not case 3

def try_case4(self, node, sibling):
  if node.parent.is_red() and sibling.are_both_children_black():
      node.parent.color = "black"
      sibling.color = "red"
      return True
  return False # not case 4

def try_case5(self, node, sibling):
  if self.is_not_none_and_red(sibling.left):
      if self.is_none_or_black(sibling.right):
          if node is node.parent.left:
              sibling.color = "red"
              sibling.left.color = "black"
              self.rotate_right(sibling)
              return True
  return False # not case 5


def try_case6(self, node, sibling):
  if self.is_none_or_black(sibling.left):
      if self.is_not_none_and_red(sibling.right):
          if node is node.parent.right:
              sibling.color = "red"
              sibling.right.color = "black"
              self.rotate_left(sibling)
              return True
  return False # not case 6