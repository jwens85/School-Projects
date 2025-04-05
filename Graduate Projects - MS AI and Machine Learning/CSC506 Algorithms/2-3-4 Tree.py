class Node:
    def __init__(self, keys=None, children=None):
        self.keys = keys or []        # List of keys in node (1, 2, or 3 keys)
        self.children = children or [] # List of children nodes (2 to 4 children)

    def is_leaf(self):
        return len(self.children) == 0

    def is_full(self):
        return len(self.keys) == 3     # Full when node has 3 keys (a 4-node)


class TwoThreeFourTree:
    def __init__(self):
        self.root = Node()

    def search(self, key, node=None):
        if node is None:
            node = self.root

        # Check if the key is in this node's keys
        for k in node.keys:
            if key == k:
                return True

        # If leaf node, key is not present
        if node.is_leaf():
            return False

        # Determine which child to go to
        if key < node.keys[0]:
            return self.search(key, node.children[0])
        elif len(node.keys) == 1 or (len(node.keys) > 1 and key < node.keys[1]):
            return self.search(key, node.children[1])
        elif len(node.keys) == 2 or (len(node.keys) > 2 and key < node.keys[2]):
            return self.search(key, node.children[2])
        else:
            return self.search(key, node.children[3])

    def insert(self, key):
        # If the root is full (3 keys), split it and create a new root
        if self.root.is_full():
            new_root = Node()
            new_root.children.append(self.root)
            self.split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key)

    def _insert_non_full(self, node, key):
        # If node is a leaf, insert key in the correct position
        if node.is_leaf():
            node.keys.append(key)
            node.keys.sort()
        else:
            # Find the child to recurse on
            i = len(node.keys) - 1
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1

            # If the child is full, split it
            if node.children[i].is_full():
                self.split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key)

    def split_child(self, parent, index):
        child = parent.children[index]
        mid_key = child.keys[1]

        # Create a new node with the last key and children of the split node
        new_child = Node(keys=[child.keys[2]], children=child.children[2:])
        child.keys = [child.keys[0]]
        child.children = child.children[:2]

        # Insert the middle key into the parent node
        parent.keys.insert(index, mid_key)
        parent.children.insert(index + 1, new_child)
