class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t = t  # Minimum degree (defines the range for the number of keys)
        self.leaf = leaf  # True if the node is a leaf
        self.keys = []  # List of keys in the node
        self.children = []  # List of children BTreeNode instances

    def is_minimal(self):
        return len(self.keys) < self.t - 1  # Node is underflowing if fewer than t-1 keys


class BTree:
    def __init__(self, t):
        self.root = BTreeNode(t, leaf=True)
        self.t = t  # Minimum degree

    def delete(self, key):
        """
        Delete a key from the B-tree, focusing on the leaf node case.
        """
        if not self.root:
            print("The tree is empty")
            return

        self._delete(self.root, key)

        # If the root has no keys and it has children, shrink the tree height
        if len(self.root.keys) == 0 and not self.root.leaf:
            self.root = self.root.children[0]

    def _delete(self, node, key):
        t = self.t

        # Find the index of the key in the current node
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        # Case 1: Key is in this node
        if i < len(node.keys) and node.keys[i] == key:
            if node.leaf:
                # Leaf Case: Key is in a leaf node, simply remove it
                node.keys.pop(i)
            else:
                # Other cases if the node is not a leaf (not covered here)
                pass
        else:
            # Case 2: Key is not in this node and we need to recurse
            if node.leaf:
                # If the key is not in a leaf, it's not in the tree
                print("The key is not in the tree")
                return

            # Recursive case: find the child to search
            if i < len(node.children) and node.children[i].is_minimal():
                self._fix_child(node, i)

            # Recursively delete from the child
            if i >= len(node.keys):
                i -= 1
            self._delete(node.children[i], key)

    def _fix_child(self, parent, index):
        """
        Ensure the child at `index` in `parent` has at least t-1 keys.
        """
        t = self.t
        child = parent.children[index]

        # Borrow from left sibling if possible
        if index > 0 and len(parent.children[index - 1].keys) >= t:
            left_sibling = parent.children[index - 1]
            child.keys.insert(0, parent.keys[index - 1])
            parent.keys[index - 1] = left_sibling.keys.pop()
            if not left_sibling.leaf:
                child.children.insert(0, left_sibling.children.pop())

        # Borrow from right sibling if possible
        elif index < len(parent.children) - 1 and len(parent.children[index + 1].keys) >= t:
            right_sibling = parent.children[index + 1]
            child.keys.append(parent.keys[index])
            parent.keys[index] = right_sibling.keys.pop(0)
            if not right_sibling.leaf:
                child.children.append(right_sibling.children.pop(0))

        # Merge with a sibling if borrowing isn’t possible
        else:
            if index < len(parent.children) - 1:
                self._merge(parent, index)
            else:
                self._merge(parent, index - 1)

    def _merge(self, parent, index):
        """
        Merge the child at index and child at index + 1 in `parent`.
        """
        child = parent.children[index]
        sibling = parent.children[index + 1]
        t = self.t

        # Pull down the separator key from the parent
        child.keys.append(parent.keys.pop(index))
        child.keys.extend(sibling.keys)

        # Add the sibling’s children if not a leaf
        if not child.leaf:
            child.children.extend(sibling.children)

        # Remove sibling from parent’s children
        parent.children.pop(index + 1)

    def search(self, key, node=None):
        """
        Search for a key in the B-tree.
        """
        if node is None:
            node = self.root

        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and node.keys[i] == key:
            return True

        if node.leaf:
            return False

        return self.search(key, node.children[i])
