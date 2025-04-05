class BSTIterator:
  def __init__(self, root):
      # Initialize with the leftmost (smallest) node
      self.current = self._leftmost_node(root)

  def _leftmost_node(self, node):
      # Helper function to find the leftmost node
      while node and node.left:
          node = node.left
      return node

  def __iter__(self):
      return self

  def __next__(self):
      if not self.current:
          raise StopIteration

      # Capture the current node's data to return
      next_value = self.current.data
      # Move to the successor
      self.current = self.current.get_successor()
      return next_value

  # Python 2 compatibility
  next = __next__
