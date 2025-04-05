class ChainingHashTable:
  def __init__(self, initial_capacity=10):
      self.capacity = initial_capacity
      self.table = [[] for _ in range(self.capacity)]

  def _hash(self, key):
      return hash(key) % self.capacity

  def insert(self, key, value):
      index = self._hash(key)
      for kvp in self.table[index]:
          if kvp[0] == key:
              kvp[1] = value
              return
      self.table[index].append([key, value])
      if self._get_load_factor() > 0.75:
          self._resize()

  def search(self, key):
      index = self._hash(key)
      bucket = self.table[index]
      for kvp in bucket:
          if kvp[0] == key:
              return kvp[1]
      return None

  def remove(self, key):
      index = self._hash(key)
      bucket = self.table[index]
      for i in range(len(bucket)):
          if bucket[i][0] == key:
              del bucket[i]
              return True
      return False

  def _resize(self):
      old_table = self.table
      self.capacity *= 2
      self.table = [[] for _ in range(self.capacity)]
      for bucket in old_table:
          for key, value in bucket:
              self.insert(key, value)

  def _get_load_factor(self):
      num_items = sum(len(bucket) for bucket in self.table)
      return num_items / self.capacity

  def __str__(self):
      items = []
      for bucket in self.table:
          items.extend([str(kvp[0]) + ": " + str(kvp[1]) for kvp in bucket])
      return "{ " + ", ".join(items) + " }"
