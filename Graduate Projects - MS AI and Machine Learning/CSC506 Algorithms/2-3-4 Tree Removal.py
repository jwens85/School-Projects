class Node:
  def __init__(self, is_leaf=True):
      self.keys = []  # List of keys in the node
      self.children = []  # List of children (empty for leaf nodes)
      self.is_leaf = is_leaf  # True if the node is a leaf

  def is_minimal(self, t):
      # Node is minimal if it has t - 1 keys
      return len(self.keys) < t

  def is_full(self, t):
      # Node is full if it has 2t - 1 keys
      return len(self.keys) == (2 * t) - 1


class TwoThreeFourTree:
  def __init__(self, t):
      self.root = Node()  # Initialize with an empty root
      self.t = t  # Minimum degree

  def remove(self, key):
      """
      Remove a key from the 2-3-4 tree.
      """
      if not self.root:
          print("Tree is empty")
          return

      self._remove(self.root, key)

      # Shrink the root if it has no keys and it's not a leaf
      if len(self.root.keys) == 0 and not self.root.is_leaf:
          self.root = self.root.children[0]

  def _remove(self, node, key):
      t = self.t

      # Case 1: Key is in the node
      if key in node.keys:
          key_index = node.keys.index(key)

          if node.is_leaf:
              # Case 1a: Key is in a leaf node, simply remove it
              node.keys.pop(key_index)
          else:
              # Case 1b: Key is in an internal node
              # Replace with predecessor or successor
              if len(node.children[key_index].keys) >= t:
                  node.keys[key_index] = self._get_predecessor(node, key_index)
                  self._remove(node.children[key_index], node.keys[key_index])
              elif len(node.children[key_index + 1].keys) >= t:
                  node.keys[key_index] = self._get_successor(node, key_index)
                  self._remove(node.children[key_index + 1], node.keys[key_index])
              else:
                  # Merge children and then delete the key
                  self._merge(node, key_index)
                  self._remove(node.children[key_index], key)

      else:
          # Case 2: Key is not in the node
          if node.is_leaf:
              # If we reach a leaf node and don't find the key
              print("Key not found in the tree")
              return

          # Find the child to recurse into
          child_index = 0
          while child_index < len(node.keys) and key > node.keys[child_index]:
              child_index += 1

          # Ensure the child has at least t keys
          if node.children[child_index].is_minimal(t):
              self._fix_child(node, child_index)

          # Recursively remove from the child
          self._remove(node.children[child_index], key)

  def _fix_child(self, parent, index):
      """
      Ensure the child at `index` in `parent` has at least t keys.
      """
      t = self.t
      child = parent.children[index]

      # Borrow from left sibling if possible
      if index > 0 and len(parent.children[index - 1].keys) >= t:
          left_sibling = parent.children[index - 1]
          child.keys.insert(0, parent.keys[index - 1])
          parent.keys[index - 1] = left_sibling.keys.pop()
          if not left_sibling.is_leaf:
              child.children.insert(0, left_sibling.children.pop())

      # Borrow from right sibling if possible
      elif index < len(parent.children) - 1 and len(parent.children[index + 1].keys) >= t:
          right_sibling = parent.children[index + 1]
          child.keys.append(parent.keys[index])
          parent.keys[index] = right_sibling.keys.pop(0)
          if not right_sibling.is_leaf:
              child.children.append(right_sibling.children.pop(0))

      # Merge with a sibling if borrowing isn’t possible
      else:
          if index < len(parent.children) - 1:
              self._merge(parent, index)
          else:
              self._merge(parent, index - 1)

  def _merge(self, parent, index):
      """
      Merge child at index and child at index + 1 in `parent`.
      """
      child = parent.children[index]
      sibling = parent.children[index + 1]
      t = self.t

      # Pull down the separator key from the parent
      child.keys.append(parent.keys.pop(index))
      child.keys.extend(sibling.keys)

      # Add the sibling’s children if not leaf
      if not child.is_leaf:
          child.children.extend(sibling.children)

      # Remove sibling from parent’s children
      parent.children.pop(index + 1)

  def _get_predecessor(self, node, index):
      """
      Get the predecessor key (largest key in left subtree).
      """
      current = node.children[index]
      while not current.is_leaf:
          current = current.children[-1]
      return current.keys[-1]

  def _get_successor(self, node, index):
      """
      Get the successor key (smallest key in right subtree).
      """
      current = node.children[index + 1]
      while not current.is_leaf:
          current = current.children[0]
      return current.keys[0]
