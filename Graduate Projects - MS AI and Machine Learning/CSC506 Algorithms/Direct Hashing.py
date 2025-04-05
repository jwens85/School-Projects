class DirectHashingHashTable:
  def __init__(self, max_key):
      # Initialize the hash table with None values up to the maximum possible key
      self.table = [None] * (max_key + 1)

  def insert(self, key, value):
      if key >= len(self.table):
          raise IndexError("Key exceeds the maximum limit of the hash table.")
      self.table[key] = value

  def search(self, key):
      if key >= len(self.table):
          raise IndexError("Key exceeds the maximum limit of the hash table.")
      return self.table[key]

  def remove(self, key):
      if key >= len(self.table):
          raise IndexError("Key exceeds the maximum limit of the hash table.")
      value = self.table[key]
      self.table[key] = None
      return value

  def __str__(self):
      items = [f"{i}: {self.table[i]}" for i in range(len(self.table)) if self.table[i] is not None]
      return "{ " + ", ".join(items) + " }"