class BTreeNode:
  def __init__(self, t, leaf=False):
      self.t = t
      self.leaf = leaf
      self.keys = []
      self.children = []

  def is_underflow(self):
      return len(self.keys) < self.t - 1  # Underflow if fewer than t-1 keys

  def rotate_left(self, parent_index):
      """
      Perform left rotation with the right sibling.
      """
      # Borrow key from the right sibling
      sibling = self.parent.children[parent_index + 1]
      self.keys.append(self.parent.keys[parent_index])  # Pull key from parent
      self.parent.keys[parent_index] = sibling.keys.pop(0)  # Move sibling's key up to parent

      # Move sibling's child to current node (if it has children)
      if not self.leaf:
          self.children.append(sibling.children.pop(0))

  def rotate_right(self, parent_index):
      """
      Perform right rotation with the left sibling.
      """
      # Borrow key from the left sibling
      sibling = self.parent.children[parent_index - 1]
      self.keys.insert(0, self.parent.keys[parent_index - 1])  # Pull key from parent
      self.parent.keys[parent_index - 1] = sibling.keys.pop()  # Move sibling's key up to parent

      # Move sibling's child to current node (if it has children)
      if not self.leaf:
          self.children.insert(0, sibling.children.pop())

  def merge_with_sibling(self, parent_index):
      """
      Merge this node with a sibling node by fusing with a parent's key.
      """
      sibling = self.parent.children[parent_index + 1] if parent_index < len(self.parent.children) - 1 else self.parent.children[parent_index - 1]

      # Merge parent's key and sibling's keys into the current node
      if parent_index < len(self.parent.children) - 1:
          self.keys.append(self.parent.keys.pop(parent_index))
          self.keys.extend(sibling.keys)
          if not self.leaf:
              self.children.extend(sibling.children)
          self.parent.children.pop(parent_index + 1)
      else:
          self.keys.insert(0, self.parent.keys.pop(parent_index - 1))
          self.keys = sibling.keys + self.keys
          if not self.leaf:
              self.children = sibling.children + self.children
          self.parent.children.pop(parent_index - 1)


class BTree:
  def __init__(self, t):
      self.root = BTreeNode(t, leaf=True)
      self.t = t

  def handle_underflow(self, node):
      """
      Handle underflow situation in a node.
      """
      parent = node.parent
      index = parent.children.index(node)

      # Check for possible rotations with siblings
      if index > 0 and len(parent.children[index - 1].keys) >= self.t:
          node.rotate_right(index)
      elif index < len(parent.children) - 1 and len(parent.children[index + 1].keys) >= self.t:
          node.rotate_left(index)
      else:
          # If rotation isn’t possible, perform fusion (merge)
          node.merge_with_sibling(index)
