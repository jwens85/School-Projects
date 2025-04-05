class HashItem:
  def __init__(self, key, value):
      self.key = key
      self.value = value

class HashTable:
  def __init__(self, size=10):
      self.size = size
      self.table = [None] * self.size

  def _hash_function(self, key):
      # Simple hash function to convert the key into a table index
      return hash(key) % self.size

  def insert(self, key, value):
      # Compute the index using the hash function
      index = self._hash_function(key)
      # Handle collisions by chaining (storing items in a list at each index)
      if self.table[index] is None:
          self.table[index] = []
      # Append the new item to the list at the computed index
      self.table[index].append(HashItem(key, value))

  def retrieve(self, key):
      # Compute the index
      index = self._hash_function(key)
      # Retrieve items at the index
      if self.table[index] is not None:
          # Iterate through items in the list at this index to find the correct item
          for item in self.table[index]:
              if item.key == key:
                  return item.value
      # If the key is not found, return None
      return None

# Creating an instance of HashTable
hash_table = HashTable(size=10)  # Initializes a hash table with 10 slots

# Inserting key-value pairs into the hash table
hash_table.insert("name", "John")
hash_table.insert("age", 30)
hash_table.insert("occupation", "Software Engineer")
hash_table.insert("city", "Sandusky")

# Retrieving values by keys
print("Name:", hash_table.retrieve("name"))  # Output: John
print("Age:", hash_table.retrieve("age"))    # Output: 30
print("Occupation:", hash_table.retrieve("occupation"))  # Output: Software Engineer
print("City:", hash_table.retrieve("city"))  # Output: Sandusky
