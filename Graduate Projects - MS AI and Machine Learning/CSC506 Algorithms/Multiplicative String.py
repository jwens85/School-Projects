class MultiplicativeStringHashTable:
  def __init__(self, capacity=10):
      self.capacity = capacity
      self.table = [None] * self.capacity

  def _multiplicative_hash(self, key):
      hash_value = 0
      prime_multiplier = 31
      for char in key:
          hash_value = (hash_value * prime_multiplier + ord(char)) % self.capacity
      return hash_value

  def insert(self, key, value):
      index = self._multiplicative_hash(key)
      # Handling collision by chaining
      if self.table[index] is None:
          self.table[index] = [(key, value)]
      else:
          # Check if the key exists, if so update the value, otherwise append
          for item in self.table[index]:
              if item[0] == key:
                  item[1] = value
                  return
          self.table[index].append((key, value))

  def search(self, key):
      index = self._multiplicative_hash(key)
      bucket = self.table[index]
      if bucket:
          for item in bucket:
              if item[0] == key:
                  return item[1]
      return None

  def remove(self, key):
      index = self._multiplicative_hash(key)
      bucket = self.table[index]
      if bucket:
          for i, item in enumerate(bucket):
              if item[0] == key:
                  removed_value = item[1]
                  del bucket[i]
                  return removed_value
      return None

  def __str__(self):
      items = [f"{kvp[0]}: {kvp[1]}" for bucket in self.table if bucket for kvp in bucket]
      return "{ " + ", ".join(items) + " }"
