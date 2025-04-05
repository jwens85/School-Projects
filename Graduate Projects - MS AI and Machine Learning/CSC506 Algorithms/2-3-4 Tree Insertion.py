class Node:
    def __init__(self, is_leaf=True):
        self.keys = []  # List of keys in the node
        self.children = []  # List of children (empty for leaf nodes)
        self.is_leaf = is_leaf  # Boolean indicating if the node is a leaf

    def is_full(self):
        # Node is full if it has 3 keys (it's a 4-node)
        return len(self.keys) == 3


class TwoThreeFourTree:
    def __init__(self):
        self.root = Node()  # Initialize with an empty root

    def split_child(self, parent, child_index):
        """
        Split a full child node into two nodes and move the middle key up to the parent.
        """
        child = parent.children[child_index]
        mid_key = child.keys[1]

        # Create a new node with the last key and children of the full child
        new_child = Node(is_leaf=child.is_leaf)
        new_child.keys = [child.keys[2]]
        if not child.is_leaf:
            new_child.children = child.children[2:]
        child.keys = [child.keys[0]]
        child.children = child.children[:2]

        # Insert the middle key into the parent and add the new child
        parent.keys.insert(child_index, mid_key)
        parent.children.insert(child_index + 1, new_child)

    def insert(self, key):
        """
        Insert a key into the 2-3-4 tree.
        """
        root = self.root

        # If the root is full, split it and create a new root
        if root.is_full():
            new_root = Node(is_leaf=False)
            new_root.children.append(root)
            self.split_child(new_root, 0)
            self.root = new_root
            self._insert_non_full(new_root, key)
        else:
            self._insert_non_full(root, key)

    def _insert_non_full(self, node, key):
        """
        Insert a key into a node that is guaranteed not to be full.
        """
        if node.is_leaf:
            # Insert the key into the leaf node
            node.keys.append(key)
            node.keys.sort()  # Keep keys sorted
        else:
            # Find the child to recurse into
            i = len(node.keys) - 1
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1

            # If the child is full, split it
            if node.children[i].is_full():
                self.split_child(node, i)
                if key > node.keys[i]:  # Adjust i if the split promotes a new key
                    i += 1

            # Recurse into the appropriate child
            self._insert_non_full(node.children[i], key)

    def search(self, key, node=None):
        """
        Search for a key in the 2-3-4 tree.
        """
        if node is None:
            node = self.root

        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            return True  # Key found
        elif node.is_leaf:
            return False  # Key not found in leaf
        else:
            return self.search(key, node.children[i])
