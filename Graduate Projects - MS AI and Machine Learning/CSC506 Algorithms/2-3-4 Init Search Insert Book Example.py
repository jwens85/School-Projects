      class Tree234:
          # Initializes the tree with the root node reference set to None.
          def __init__(self):
              self.root = None

          # Inserts a new key into this tree, provided the tree doesn't already
          # contain the same key.
          def insert(self, key, node = None, node_parent = None):
              # Special case for empty tree
              if self.root == None:
                  self.root = Node234(key)
                  return self.root

              # If the node argument is null, recursively call with root
              if node == None:
                  return self.insert(key, self.root, None)

              # Check for duplicate key
              if node.has_key(key):
                  # Duplicate keys are not allowed
                  return None

              # Preemptively split full nodes
              if node.C != None:
                  node = self.split(node, node_parent)

              if not node.is_leaf():
                  if key < node.A:
                      return self.insert(key, node.left, node)
                  elif node.B == None or key < node.B:
                      return self.insert(key, node.middle1, node)
                  elif node.C == None or key < node.C:
                      return self.insert(key, node.middle2, node)
                  else:
                      return self.insert(key, node.right, node)

              # key can be inserted into leaf node
              node.insert_key(key)
              return node

          # Searches this tree for the specified key. If found, the node containing
          # the key is returned. Otherwise None is returned.
          def search(self, key):
              return self.search_recursive(key, self.root)

          # Recursive helper method for search.
          def search_recursive(self, key, node):
              if node == None:
                  return None

              # Check if the node contains the key
              if node.has_key(key):
                  return node

              # Recursively search the appropriate subtree
              if key < node.A:
                  return self.search_recursive(key, node.left)
              elif node.B == None or key < node.B:
                  return self.search_recursive(key, node.middle1)
              elif node.C == None or key < node.C:
                  return self.search_recursive(key, node.middle2)
              return self.search_recursive(key, node.right)

          # Splits a full node, moving the middle key up into the parent node.
          def split(self, node, node_parent):
              split_left = Node234(node.A, node.left, node.middle1)
              split_right = Node234(node.C, node.middle2, node.right)
              if node_parent is not None:
                  node_parent.insert_key_with_children(node.B, split_left, split_right)
              else:
                  # Split root
                  node_parent = Node234(node.B, split_left, split_right)
                  self.root = node_parent

              return node_parent