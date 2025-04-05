class BTreeNode:
  def __init__(self, t, leaf=False):
      self.t = t  # Minimum degree (defines the range for the number of keys)
      self.leaf = leaf  # True if the node is a leaf
      self.keys = []  # List of keys in the node
      self.children = []  # List of children BTreeNode instances

  def insert_non_full(self, key):
      i = len(self.keys) - 1

      # If this node is a leaf, insert the new key in the correct position
      if self.leaf:
          self.keys.append(0)
          while i >= 0 and key < self.keys[i]:
              self.keys[i + 1] = self.keys[i]
              i -= 1
          self.keys[i + 1] = key
      else:
          # Find the child to recurse into
          while i >= 0 and key < self.keys[i]:
              i -= 1
          i += 1

          # Split the child if it's full
          if len(self.children[i].keys) == (2 * self.t) - 1:
              self.split_child(i)
              if key > self.keys[i]:
                  i += 1
          self.children[i].insert_non_full(key)

  def split_child(self, i):
      t = self.t
      y = self.children[i]
      z = BTreeNode(t, y.leaf)

      # Move the last (t-1) keys of y to z
      z.keys = y.keys[t: (2 * t) - 1]
      y.keys = y.keys[0: t - 1]

      # Move the last t children of y to z
      if not y.leaf:
          z.children = y.children[t: 2 * t]
          y.children = y.children[0: t]

      # Insert the middle key of y into this node
      self.children.insert(i + 1, z)
      self.keys.insert(i, y.keys.pop(-1))

  def search(self, key):
      i = 0
      while i < len(self.keys) and key > self.keys[i]:
          i += 1

      # If the found key is equal to the key
      if i < len(self.keys) and self.keys[i] == key:
          return self

      # If the key is not found and this is a leaf node
      if self.leaf:
          return None

      # Go to the appropriate child
      return self.children[i].search(key)

class BTree:
  def __init__(self, t):
      self.root = BTreeNode(t, True)
      self.t = t

  def insert(self, key):
      root = self.root
      if len(root.keys) == (2 * self.t) - 1:
          new_root = BTreeNode(self.t, False)
          new_root.children.append(root)
          new_root.split_child(0)
          i = 0
          if new_root.keys[0] < key:
              i += 1
          new_root.children[i].insert_non_full(key)
          self.root = new_root
      else:
          root.insert_non_full(key)

  def search(self, key):
      return self.root.search(key)
