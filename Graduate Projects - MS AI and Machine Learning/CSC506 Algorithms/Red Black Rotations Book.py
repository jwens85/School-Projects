class RedBlackTree:
  def __init__(self):
      self.root = None

  def __len__(self):
      if self.root is None:
          return 0
      return self.root.count()

  def insert(self, key):
      new_node = RBTNode(key, None, True, None, None)
      self.insert_node(new_node)

  def insert_node(self, node):
      # Begin with normal BST insertion
      if self.root is None:
          # Special case for root
          self.root = node
      else:
          current_node = self.root
          while current_node is not None:
              if node.key < current_node.key:
                  if current_node.left is None:
                      current_node.set_child("left", node)
                      break
                  else:
                      current_node = current_node.left
              else:
                  if current_node.right is None:
                      current_node.set_child("right", node)
                      break
                  else:
                      current_node = current_node.right

      # Color the node red
      node.color = "red"

      # Balance
      self.insertion_balance(node)

  def insertion_balance(self, node):
      # If node is the tree's root, then color node black and return
      if node.parent is None:
          node.color = "black"
          return

      # If parent is black, then return without any alterations
      if node.parent.is_black():
          return

      # References to parent, grandparent, and uncle are needed for remaining operations
      parent = node.parent
      grandparent = node.get_grandparent()
      uncle = node.get_uncle()

      # If parent and uncle are both red, then color parent and uncle black, color grandparent
      # red, recursively balance  grandparent, then return
      if uncle is not None and uncle.is_red():
          parent.color = uncle.color = "black"
          grandparent.color = "red"
          self.insertion_balance(grandparent)
          return

      # If node is parent's right child and parent is grandparent's left child, then rotate left
      # at parent, update node and parent to point to parent and grandparent, respectively
      if node is parent.right and parent is grandparent.left:
          self.rotate_left(parent)
          node = parent
          parent = node.parent
      # Else if node is parent's left child and parent is grandparent's right child, then rotate
      # right at parent, update node and parent to point to parent and grandparent, respectively
      elif node is parent.left and parent is grandparent.right:
          self.rotate_right(parent)
          node = parent
          parent = node.parent

      # Color parent black and grandparent red
      parent.color = "black"
      grandparent.color = "red"

      # If node is parent's left child, then rotate right at grandparent, otherwise rotate left
      # at grandparent
      if node is parent.left:
          self.rotate_right(grandparent)
      else:
          self.rotate_left(grandparent)

  def rotate_left(self, node):
      right_left_child = node.right.left
      if node.parent != None:
          node.parent.replace_child(node, node.right)
      else: # node is root
          self.root = node.right
          self.root.parent = None
      node.right.set_child("left", node)
      node.set_child("right", right_left_child)

  def rotate_right(self, node):
      left_right_child = node.left.right
      if node.parent != None:
          node.parent.replace_child(node, node.left)
      else: # node is root
          self.root = node.left
          self.root.parent = None
      node.left.set_child("right", node)
      node.set_child("left", left_right_child)