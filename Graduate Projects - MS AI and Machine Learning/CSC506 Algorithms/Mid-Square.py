class MidSquareHashTable:
  def __init__(self, capacity=10):
      self.capacity = capacity
      self.table = [None] * self.capacity

  def _mid_square_hash(self, key):
      # Square the key
      squared_key = key * key
      # Convert to string to extract the middle digits
      squared_key_str = str(squared_key)
      # Extract the middle portion of the squared key string
      length = len(squared_key_str)
      mid_start = length // 2 - 1
      mid_end = mid_start + 2
      if mid_start < 0:
          mid_start = 0
          mid_end = 1
      # Convert the middle digits back to integer
      middle_digits = int(squared_key_str[mid_start:mid_end])
      # Modulo operation to fit the hash table size
      return middle_digits % self.capacity

  def insert(self, key, value):
      index = self._mid_square_hash(key)
      # Handling collision by chaining (linked list or a simple list in this case)
      if self.table[index] is None:
          self.table[index] = [(key, value)]
      else:
          # Check if the key exists, if so update the value, otherwise append
          for item in self.table[index]:
              if item[0] == key:
                  item = (key, value)
                  return
          self.table[index].append((key, value))

  def search(self, key):
      index = self._mid_square_hash(key)
      if self.table[index] is not None:
          for item in self.table[index]:
              if item[0] == key:
                  return item[1]
      return None

  def remove(self, key):
      index = self._mid_square_hash(key)
      if self.table[index] is not None:
          for i, item in enumerate(self.table[index]):
              if item[0] == key:
                  del self.table[index][i]
                  return True
      return False

  def __str__(self):
      items = [str(item) for bucket in self.table if bucket is not None for item in bucket]
      return "{ " + ", ".join(items) + " }"
