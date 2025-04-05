class ChainingHashTable:
  def __init__(self, initial_capacity=10):
      self.table = [[] for _ in range(initial_capacity)]

  def insert(self, item):
      bucket = hash(item) % len(self.table)
      if item not in self.table[bucket]:
          self.table[bucket].append(item)

  def search(self, key):
      bucket = hash(key) % len(self.table)
      return key if key in self.table[bucket] else None

  def remove(self, key):
      bucket = hash(key) % len(self.table)
      if key in self.table[bucket]:
          self.table[bucket].remove(key)
          return True
      return False
