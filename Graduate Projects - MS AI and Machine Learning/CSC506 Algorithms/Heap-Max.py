class MaxHeap:
  def __init__(self):
      self.heap = []

  def insert(self, element):
      # Add the element to the end of the heap
      self.heap.append(element)
      # Move the new element up to its proper position
      self._percolate_up(len(self.heap) - 1)

  def _percolate_up(self, index):
      # Get the parent index
      parent_index = (index - 1) // 2
      # Keep moving the element up until we find the correct position
      while index > 0 and self.heap[parent_index] < self.heap[index]:
          self.heap[parent_index], self.heap[index] = self.heap[index], self.heap[parent_index]
          index = parent_index
          parent_index = (index - 1) // 2

  def delete_max(self):
      if not self.heap:
          return None
      # Replace the root of the heap with the last element
      max_value = self.heap[0]
      self.heap[0] = self.heap[-1]
      self.heap.pop()
      # Move the new root down to the correct position
      self._percolate_down(0)
      return max_value

  def _percolate_down(self, index):
      child_index = 2 * index + 1
      while child_index < len(self.heap):
          # Check if there is a right child and it's larger than the left child
          if child_index + 1 < len(self.heap) and self.heap[child_index + 1] > self.heap[child_index]:
              child_index += 1
          # If the parent is smaller than the largest child, swap them
          if self.heap[index] < self.heap[child_index]:
              self.heap[index], self.heap[child_index] = self.heap[child_index], self.heap[index]
              index = child_index
              child_index = 2 * index + 1
          else:
              break

  def get_max(self):
      return self.heap[0] if self.heap else None

  def __str__(self):
      return str(self.heap)

# Example usage
if __name__ == "__main__":
  heap = MaxHeap()
  heap.insert(10)
  heap.insert(20)
  heap.insert(15)
  heap.insert(30)
  heap.insert(40)

  print("Heap:", heap)
  print("Max:", heap.get_max())
  print("Deleted Max:", heap.delete_max())
  print("Heap after deletion:", heap)
