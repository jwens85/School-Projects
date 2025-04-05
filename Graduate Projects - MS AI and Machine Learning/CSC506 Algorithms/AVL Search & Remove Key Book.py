# Searches for a node with a matching key. Does a regular
# binary search tree search operation. Returns the node with the
# matching key if it exists in the tree, or None if there is no
# matching key in the tree.
def search(self, key):
    current_node = self.root
    while current_node is not None:
        # Compare the current node's key with the target key.
        # If it is a match, return the current key; otherwise go
        # either to the left or right, depending on whether the 
        # current node's key is smaller or larger than the target key.
        if current_node.key == key: return current_node
        elif current_node.key < key: current_node = current_node.right
        else: current_node = current_node.left

# Attempts to remove a node with a matching key. If no node has a matching key
# then nothing is done and False is returned; otherwise the node is removed and
# True is returned.
def remove_key(self, key):
    node = self.search(key)
    if node is None:
        return False
    else:
        return self.remove_node(node)