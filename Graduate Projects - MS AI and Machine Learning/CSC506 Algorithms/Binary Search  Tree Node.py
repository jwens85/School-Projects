class BSTNode:
  def __init__(self, data, parent=None, left=None, right=None):
      self.data = data
      self.left = left
      self.right = right
      self.parent = parent

  def count(self):
      left_count = self.left.count() if self.left else 0
      right_count = self.right.count() if self.right else 0
      return 1 + left_count + right_count

  def get_successor(self):
      if self.right:
          successor = self.right
          while successor.left:
              successor = successor.left
          return successor

      node = self
      while node.parent and node == node.parent.right:
          node = node.parent
      return node.parent

  def replace_child(self, current_child, new_child):
      if current_child is self.left:
          self.left = new_child
      elif current_child is self.right:
          self.right = new_child

      if new_child:
          new_child.parent = self

  def insert(self, value):
      if value < self.data:
          if self.left:
              self.left.insert(value)
          else:
              self.left = BSTNode(value, parent=self)
      elif value > self.data:
          if self.right:
              self.right.insert(value)
          else:
              self.right = BSTNode(value, parent=self)

  def find(self, value):
      if value == self.data:
          return self
      elif value < self.data and self.left:
          return self.left.find(value)
      elif value > self.data and self.right:
          return self.right.find(value)
      return None

  def delete(self, value):
      node = self.find(value)
      if node:
          if node.left and node.right:
              successor = node.get_successor()
              node.data = successor.data
              successor.parent.replace_child(successor, successor.right)
          elif node.left:
              self.replace_child(node, node.left)
          elif node.right:
              self.replace_child(node, node.right)
          else:
              self.replace_child(node, None)
