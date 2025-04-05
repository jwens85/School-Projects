class Node234:
  # Appends 1 key and 1 child to this node.
  # Preconditions:
  # 1. This node has 1 or 2 keys
  # 2. key > all keys in this node
  # 3. Child subtree contains only keys > key
  def append_key_and_child(self, key, child):
      if self.B == None:
          self.B = key
          self.middle2 = child
      else:
          self.C = key
          self.right = child

  # Returns the number of keys in this node, which will be 1, 2, or 3.
  def count_keys(self):
      if self.C != None:
          return 3
      elif self.B != None:
          return 2
      return 1

  # Returns True if this node has the specified key, False otherwise.
  def has_key(self, key):
      return self.A == key or self.B == key or self.C == key

  # Inserts a new key into the proper location in this node.
  # Precondition: This node is a leaf and has 2 or fewer keys
  def insert_key(self, key):
      if key < self.A:
          self.C = self.B
          self.B = self.A
          self.A = key
      elif self.B == None or key < self.B:
          self.C = self.B
          self.B = key
      else:
          self.C = key

  # Inserts a new key into the proper location in this node, and
  # sets the children on either side of the inserted key.
  # Precondition: This node has 2 or fewer keys
  def insert_key_with_children(self, key, leftChild, rightChild):
      if key < self.A:
          self.C = self.B
          self.B = self.A
          self.A = key
          self.right = self.middle2
          self.middle2 = self.middle1
          self.middle1 = rightChild
          self.left = leftChild
      elif self.B == None or key < self.B:
          self.C = self.B
          self.B = key
          self.right = self.middle2
          self.middle2 = rightChild
          self.middle1 = leftChild
      else:
          self.C = key
          self.right = rightChild
          self.middle2 = leftChild

  # Returns True if this node is a leaf, False otherwise.
  def is_leaf(self):
      return self.left == None

  # Returns the child of this node that would be visited next in the
  # traversal to search for the specified key
  def next_node(self, key):
      if key < self.A:
          return self.left
      elif self.B == None or key < self.B:
          return self.middle1
      elif self.C == None or key < self.C:
          return self.middle2
      return self.right

  # Removes key A, B, or C from this node, if key_index is 0, 1, or 2,
  # respectively. Other keys and children are shifted as necessary.
  def remove_key(self, key_index):
      if key_index == 0:
          self.A = self.B
          self.B = self.C
          self.C = None
          self.left = self.middle1
          self.middle1 = self.middle2
          self.middle2 = self.right
          self.right = None
      elif key_index == 1:
          self.B = self.C
          self.C = None
          self.middle2 = self.right
          self.right = None
      elif key_index == 2:
          self.C = None
          self.right = None

  # Removes and returns the rightmost child. Two possible cases exist:
  # 1. If this node has a right child, right is set to None, and the
  #    previous right value is returned.
  # 2. Else if this node has a middle2 child, middle2 is set to None, and
  #    the previous right value is returned.
  # 3. Otherwise no action is taken, and None is returned.
  # No keys are changed in any case.
  def remove_rightmost_child(self):
      removed = None
      if self.right != None:
          removed = self.right
          self.right = None
      elif self.middle2 != None:
          removed = self.middle2
          self.middle2 = None
      return removed

  # Removes and returns the rightmost key. Three possible cases exist:
  # 1. If this node has 3 keys, C is set to None and the previous C value is returned.
  # 2. If this node has 2 keys, B is set to None and the previous B value is returned.
  # 3. Otherwise no action is taken and None is returned.
  # No children are changed in any case.
  def remove_rightmost_key(self):
      removed = None
      if self.C != None:
          removed = self.C
          self.C = None
      elif self.B != None:
          removed = self.B
          self.B = None
      return removed

  # Sets the left, middle1, middle2, or right child if the child_index
  # argument is 0, 1, 2, or 3, respectively.
  # Does nothing if the child_index argument is < 0 or > 3.
  def set_child(self, child, child_index):
      if child_index == 0:
          self.left = child
      elif child_index == 1:
          self.middle1 = child
      elif child_index == 2:
          self.middle2 = child
      elif child_index == 3:
          self.right = child

  # Sets this node's A, B, or C key if the key_index argument is 0, 1, or
  # 2, respectively.
  # Does nothing if the key_index argument is < 0 or > 2.
  def set_key(self, key, key_index):
      if key_index == 0:
          self.A = key
      elif key_index == 1:
          self.B = key
      elif key_index == 2:
          self.C = key