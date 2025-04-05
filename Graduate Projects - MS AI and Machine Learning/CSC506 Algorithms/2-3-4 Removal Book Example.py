      class Tree234:
          # Fuses a parent node and two children into one node. 
          # Precondition: Each of the three nodes must have one key each.
          def fuse(self, parent, left_node, right_node):
              if parent is self.root and parent.count_keys() == 1:
                  return self.fuse_root()

              left_node_index = parent.get_child_index(left_node)
              middle_key = parent.get_key(left_node_index)
              fused_node = Node234(left_node.A)
              fused_node.B = middle_key
              fused_node.C = right_node.A
              fused_node.left = left_node.left
              fused_node.middle1 = left_node.middle1
              fused_node.middle2 = right_node.left
              fused_node.right = right_node.middle1
              key_index = parent.get_key_index(middle_key)
              parent.remove_key(key_index)
              parent.set_child(fused_node, key_index)
              return fused_node

          # Fuses the tree's root node with the root's two children. 
          # Precondition: Each of the three nodes must have one key each.
          def fuse_root(self):        
              old_left = self.root.left
              old_middle1 = self.root.middle1
              self.root.B = self.root.A
              self.root.A = old_left.A
              self.root.C = old_middle1.A
              self.root.left = old_left.left
              self.root.middle1 = old_left.middle1
              self.root.middle2 = old_middle1.left
              self.root.right = old_middle1.middle1
              return self.root

          # Searches for, and returns, the minimum key in a subtree
          def get_min_key(self, node):
              current = node
              while current.left != None:
                  current = current.left
              return current.A

          # Finds and replaces one key with another. The replacement key must
          # be known to be a key that can be used as a replacement without violating
          # any of the 2-3-4 tree rules.
          def key_swap(self, node, existing, replacement):
              if node == None:
                  return False

              key_index = node.get_key_index(existing)
              if key_index == -1:
                  next = node.next_node(existing)
                  return self.key_swap(next, existing, replacement)

              if key_index == 0:
                  node.A = replacement
              elif key_index == 1:
                  node.B = replacement
              else:
                  node.C = replacement

              return True

          # Rotates or fuses to add 1 or 2 additional keys to a node with 1 key.
          def merge(self, node, node_parent):
              # Get references to node's siblings
              node_index = node_parent.get_child_index(node)
              left_sibling = node_parent.get_child(node_index - 1)
              right_sibling = node_parent.get_child(node_index + 1)

              # Check siblings for a key that can be transferred
              if left_sibling != None and left_sibling.count_keys() >= 2:
                  self.rotate_right(left_sibling, node_parent)
              elif right_sibling != None and right_sibling.count_keys() >= 2:
                  self.rotate_left(right_sibling, node_parent)
              else: # fuse
                  if left_sibling == None:
                      node = self.fuse(node_parent, node, right_sibling)
                  else:
                      node = self.fuse(node_parent, left_sibling, node)

              return node

          # Finds and removes the specified key from this tree.
          def remove(self, key):
              # Special case for tree with 1 key
              if self.root.is_leaf() and self.root.count_keys() == 1:
                  if self.root.A == key:
                      self.root = None
                      return True
                  return False

              current_parent = None
              current = self.root
              while current != None:
                  # Merge any non-root node with 1 key
                  if current.count_keys() == 1 and current is not self.root:
                      current = self.merge(current, current_parent)

                  # Check if current node contains key
                  key_index = current.get_key_index(key)
                  if key_index != -1:
                      if current.is_leaf():
                          current.remove_key(key_index)
                          return True

                      # The node contains the key and is not a leaf, so the key is
                      # replaced with the successor
                      tmp_child = current.get_child(key_index + 1)
                      tmp_key = self.get_min_key(tmp_child)
                      self.remove(tmp_key)
                      self.key_swap(self.root, key, tmp_key)
                      return True

                  # Current node does not contain key, so continue down tree
                  current_parent = current
                  current = current.next_node(key)

              # key not found
              return False

          def rotate_left(self, node, node_parent):
              # Get the node's left sibling
              node_index = node_parent.get_child_index(node)
              left_sibling = node_parent.get_child(node_index - 1)

              # Get the key from the parent that will be copied into the left sibling
              key_for_left_sibling = node_parent.get_key(node_index - 1)

              # Append the key to the left sibling
              left_sibling.append_key_and_child(key_for_left_sibling, node.left)

              # Replace the parent's key that was appended to the left sibling
              node_parent.set_key(node.A, node_index - 1)

              # Remove key A and left child from node
              node.remove_key(0)

          def rotate_right(self, node, node_parent):
              # Get the node's right sibling
              node_index = node_parent.get_child_index(node)
              right_sibling = node_parent.get_child(node_index + 1)

              # Get the key from the parent that will be copied into the right sibling
              key_for_right_sibling = node_parent.get_key(node_index)

              # Shift key and child references in right sibling
              right_sibling.C = right_sibling.B
              right_sibling.B = right_sibling.A
              right_sibling.right = right_sibling.middle2
              right_sibling.middle2 = right_sibling.middle1
              right_sibling.middle1 = right_sibling.left

              # Set key A and the left child of right_sibling
              right_sibling.A = key_for_right_sibling
              right_sibling.left = node.remove_rightmost_child()

              # Replace the parent's key that was prepended to the right sibling
              node_parent.set_key(node.remove_rightmost_key(), node_index)