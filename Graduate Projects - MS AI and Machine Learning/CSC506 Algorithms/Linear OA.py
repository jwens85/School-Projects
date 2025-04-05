class LinearProbingHashTable:
  def __init__(self, initial_capacity=10):
      self.capacity = initial_capacity
      self.keys = [None] * self.capacity
      self.values = [None] * self.capacity
      self.size = 0

  def _hash(self, key):
      return hash(key) % self.capacity

  def insert(self, key, value):
      index = self._hash(key)
      initial_index = index
      while self.keys[index] is not None:
          if self.keys[index] == key:
              self.values[index] = value
              return
          index = (index + 1) % self.capacity
          if index == initial_index:
              raise Exception("HashTable is full")
      self.keys[index] = key
      self.values[index] = value
      self.size += 1
      if self.size / self.capacity > 0.75:
          self._resize()

  def search(self, key):
      index = self._hash(key)
      initial_index = index
      while self.keys[index] is not None:
          if self.keys[index] == key:
              return self.values[index]
          index = (index + 1) % self.capacity
          if index == initial_index:
              break
      return None

  def remove(self, key):
      index = self._hash(key)
      initial_index = index
      while self.keys[index] is not None:
          if self.keys[index] == key:
              self.keys[index] = None
              value = self.values[index]
              self.values[index] = None
              self.size -= 1
              self._rehash_from(index)
              return value
          index = (index + 1) % self.capacity
          if index == initial_index:
              break
      return None

  def _rehash_from(self, start_index):
      index = (start_index + 1) % self.capacity
      while self.keys[index] is not None:
          key = self.keys[index]
          value = self.values[index]
          self.keys[index] = None
          self.values[index] = None
          self.size -= 1
          self.insert(key, value)
          index = (index + 1) % self.capacity

  def _resize(self):
      old_keys = self.keys
      old_values = self.values
      self.capacity *= 2
      self.keys = [None] * self.capacity
      self.values = [None] * self.capacity
      self.size = 0
      for key, value in zip(old_keys, old_values):
          if key is not None:
              self.insert(key, value)

  def __str__(self):
      items = []
      for key, value in zip(self.keys, self.values):
          if key is not None:
              items.append(f"{key}: {value}")
      return "{ " + ", ".join(items) + " }"
