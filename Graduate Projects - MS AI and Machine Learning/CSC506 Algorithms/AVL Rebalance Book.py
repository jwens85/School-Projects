class AVLTree:
  # Constructor to create an empty AVLTree. There is only
  # one data member, the tree's root Node, and it starts
  # out as None.
  def __init__(self):
      self.root = None

  # Performs a left rotation at the given node. Returns the
  # new root of the subtree.
  def rotate_left(self, node):
      # Define a convenience pointer to the right child of the 
      # left child.
      right_left_child = node.right.left

      # Step 1 - the right child moves up to the node's position.
      # This detaches node from the tree, but it will be reattached
      # later.
      if node.parent is not None:
          node.parent.replace_child(node, node.right)
      else:  # node is root
          self.root = node.right
          self.root.parent = None

      # Step 2 - the node becomes the left child of what used
      # to be its right child, but is now its parent. This will
      # detach right_left_child from the tree.
      node.right.set_child('left', node)

      # Step 3 - reattach right_left_child as the right child of node.
      node.set_child('right', right_left_child)

      return node.parent

  # Performs a right rotation at the given node. Returns the
  # subtree's new root.
  def rotate_right(self, node):
      # Define a convenience pointer to the left child of the 
      # right child.
      left_right_child = node.left.right

      # Step 1 - the left child moves up to the node's position.
      # This detaches node from the tree, but it will be reattached
      # later.
      if node.parent is not None:
          node.parent.replace_child(node, node.left)
      else:  # node is root
          self.root = node.left
          self.root.parent = None

      # Step 2 - the node becomes the right child of what used
      # to be its left child, but is now its parent. This will
      # detach left_right_child from the tree.
      node.left.set_child('right', node)

      # Step 3 - reattach left_right_child as the left child of node.
      node.set_child('left', left_right_child)

      return node.parent

  # Updates the given node's height and rebalances the subtree if
  # the balancing factor is now -2 or +2. Rebalancing is done by
  # performing a rotation. Returns the subtree's new root if
  # a rotation occurred, or the node if no rebalancing was required.
  def rebalance(self, node):

      # First update the height of this node.
      node.update_height()        

      # Check for an imbalance.
      if node.get_balance() == -2:

          # The subtree is too big to the right.
          if node.right.get_balance() == 1:
              # Double rotation case. First do a right rotation
              # on the right child.
              self.rotate_right(node.right)

          # A left rotation will now make the subtree balanced.
          return self.rotate_left(node)

      elif node.get_balance() == 2:

          # The subtree is too big to the left
          if node.left.get_balance() == -1:
              # Double rotation case. First do a left rotation
              # on the left child.
              self.rotate_left(node.left)

          # A right rotation will now make the subtree balanced.
          return self.rotate_right(node)

      # No imbalance, so just return the original node.
      return node