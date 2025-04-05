def remove_node(self, node):
  # Base case: 
  if node is None:
      return False

  # Parent needed for rebalancing.
  parent = node.parent

  # Case 1: Internal node with 2 children
  if node.left is not None and node.right is not None:
      # Find successor
      successor_node = node.right
      while successor_node.left != None:
          successor_node = successor_node.left

      # Copy the value from the node
      node.key = successor_node.key

      # Recursively remove successor
      self.remove_node(successor_node)

      # Nothing left to do since the recursive call will have rebalanced
      return True

  # Case 2: Root node (with 1 or 0 children)
  elif node is self.root:
      if node.left is not None:
           self.root = node.left
      else:
           self.root = node.right

      if self.root is not None:
           self.root.parent = None

      return True

  # Case 3: Internal with left child only
  elif node.left is not None:
      parent.replace_child(node, node.left)

  # Case 4: Internal with right child only OR leaf
  else:
      parent.replace_child(node, node.right)

  # node is gone. Anything that was below node that has persisted is already correctly
  # balanced, but ancestors of node may need rebalancing.
  node = parent
  while node is not None:
      self.rebalance(node)            
      node = node.parent

  return True