class DoubleHashingHashTable:
  def __init__(self, initial_capacity=10):
      self.capacity = initial_capacity
      self.keys = [None] * self.capacity
      self.values = [None] * self.capacity
      self.size = 0

  def _hash1(self, key):
      return hash(key) % self.capacity

  def _hash2(self, key):
      # Secondary hash function: must be non-zero
      return 1 + (hash(key) % (self.capacity - 1))

  def insert(self, key, value):
      if self.size / self.capacity >= 0.75:
          self._resize()
      index = self._hash1(key)
      if self.keys[index] is None:
          self.keys[index] = key
          self.values[index] = value
          self.size += 1
      else:
          step = self._hash2(key)
          i = 1
          while True:
              new_index = (index + i * step) % self.capacity
              if self.keys[new_index] is None:
                  self.keys[new_index] = key
                  self.values[new_index] = value
                  self.size += 1
                  break
              i += 1

  def search(self, key):
      index = self._hash1(key)
      step = self._hash2(key)
      i = 0
      while self.keys[(index + i * step) % self.capacity] is not None:
          if self.keys[(index + i * step) % self.capacity] == key:
              return self.values[(index + i * step) % self.capacity]
          i += 1
      return None

  def remove(self, key):
      index = self._hash1(key)
      step = self._hash2(key)
      i = 0
      while self.keys[(index + i * step) % self.capacity] is not None:
          if self.keys[(index + i * step) % self.capacity] == key:
              self.keys[(index + i * step) % self.capacity] = None
              value = self.values[(index + i * step) % self.capacity]
              self.values[(index + i * step) % self.capacity] = None
              self.size -= 1
              self._rehash_from((index + i * step) % self.capacity)
              return value
          i += 1
      return None

  def _rehash_from(self, start_index):
      original_index = start_index
      step = self._hash2(self.keys[start_index])
      i = 1
      while self.keys[(original_index + i * step) % self.capacity] is not None:
          key = self.keys[(original_index + i * step) % self.capacity]
          value = self.values[(original_index + i * step) % self.capacity]
          self.keys[(original_index + i * step) % self.capacity] = None
          self.values[(original_index + i * step) % self.capacity] = None
          self.size -= 1
          self.insert(key, value)
          i += 1

  def _resize(self):
      old_keys = self.keys[:]
      old_values = self.values[:]
      self.capacity *= 2
      self.keys = [None] * self.capacity
      self.values = [None] * self.capacity
      self.size = 0
      for k, v in zip(old_keys, old_values):
          if k is not None:
              self.insert(k, v)

  def __str__(self):
      items = [(str(k) + ": " + str(v)) for k, v in zip(self.keys, self.values) if k is not None]
      return "{ " + ", ".join(items) + " }"
