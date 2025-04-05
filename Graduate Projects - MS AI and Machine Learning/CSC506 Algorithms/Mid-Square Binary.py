class BinaryMidSquareHashTable:
  def __init__(self, capacity=10):
      self.capacity = capacity
      self.table = [None] * self.capacity

  def _binary_mid_square_hash(self, key):
      # Square the key
      squared_key = key * key
      # Convert the squared key to binary
      squared_key_binary = bin(squared_key)[2:]  # remove the '0b' prefix
      # Calculate the number of bits to extract
      num_bits = len(bin(self.capacity)[2:])  # number of bits to index all buckets
      mid_start = len(squared_key_binary) // 2 - num_bits // 2
      mid_end = mid_start + num_bits
      # Extract the middle bits and convert back to integer
      middle_bits = squared_key_binary[mid_start:mid_end]
      # If extracted bits are shorter than needed, pad with zeros
      if len(middle_bits) < num_bits:
          middle_bits = '0' * (num_bits - len(middle_bits)) + middle_bits
      # Convert to decimal integer
      middle_bits_decimal = int(middle_bits, 2)
      # Modulo operation to fit the hash table size
      return middle_bits_decimal % self.capacity

  def insert(self, key, value):
      index = self._binary_mid_square_hash(key)
      # Handling collision by chaining
      if self.table[index] is None:
          self.table[index] = [(key, value)]
      else:
          for item in self.table[index]:
              if item[0] == key:
                  item = (key, value)
                  return
          self.table[index].append((key, value))

  def search(self, key):
      index = self._binary_mid_square_hash(key)
      if self.table[index] is not None:
          for item in self.table[index]:
              if item[0] == key:
                  return item[1]
      return None

  def remove(self, key):
      index = self._binary_mid_square_hash(key)
      if self.table[index] is not None:
          for i, item in enumerate(self.table[index]):
              if item[0] == key:
                  del self.table[index][i]
                  return True
      return False

  def __str__(self):
      items = [str(item) for bucket in self.table if bucket is not None for item in bucket]
      return "{ " + ", ".join(items) + " }"
