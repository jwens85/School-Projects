def insert(self, node):

# Special case: if the tree is empty, just set the root to
# the new node.
if self.root is None:
  self.root = node
  node.parent = None

else:
  # Step 1 - do a regular binary search tree insert.
  current_node = self.root
  while current_node is not None:
      # Choose to go left or right
      if node.key < current_node.key:
          # Go left. If left child is None, insert the new
          # node here.
          if current_node.left is None:
              current_node.left = node
              node.parent = current_node
              current_node = None
          else:
              # Go left and do the loop again.
              current_node = current_node.left
      else:
          # Go right. If the right child is None, insert the
          # new node here.
          if current_node.right is None:
              current_node.right = node
              node.parent = current_node
              current_node = None
          else:
              # Go right and do the loop again.
              current_node = current_node.right

  # Step 2 - Rebalance along a path from the new node's parent up
  # to the root.
  node = node.parent
  while node is not None:
      self.rebalance(node)
      node = node.parent