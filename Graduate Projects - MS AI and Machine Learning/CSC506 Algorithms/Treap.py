import random

class TreapNode:
    def __init__(self, key, priority=None):
        # Initialize a node with a given key and priority.
        # If no priority is provided, assign a random priority.
        self.key = key
        self.priority = priority if priority is not None else random.randint(1, 100)
        self.left = None   # Left child node
        self.right = None  # Right child node

class Treap:
    def __init__(self):
        # Initialize an empty Treap with root as None.
        self.root = None

    def _rotate_right(self, node):
        # Perform a right rotation around the given node.
        new_root = node.left           # The left child becomes the new root.
        node.left = new_root.right     # The right subtree of new_root becomes the left subtree of node.
        new_root.right = node          # The original node becomes the right child of new_root.
        return new_root                # Return the new root after rotation.

    def _rotate_left(self, node):
        # Perform a left rotation around the given node.
        new_root = node.right          # The right child becomes the new root.
        node.right = new_root.left     # The left subtree of new_root becomes the right subtree of node.
        new_root.left = node           # The original node becomes the left child of new_root.
        return new_root                # Return the new root after rotation.

    def _insert(self, node, key, priority):
        # Recursive helper function to insert a key with a given priority.
        if node is None:
            # If the current node is None, create a new node.
            return TreapNode(key, priority)

        if key < node.key:
            # Insert into the left subtree.
            node.left = self._insert(node.left, key, priority)
            # Rotate right if heap property is violated.
            if node.left and node.left.priority > node.priority:
                node = self._rotate_right(node)
        elif key > node.key:
            # Insert into the right subtree.
            node.right = self._insert(node.right, key, priority)
            # Rotate left if heap property is violated.
            if node.right and node.right.priority > node.priority:
                node = self._rotate_left(node)
        else:
            # If the key already exists, do nothing (no duplicates allowed).
            pass

        return node  # Return the (possibly updated) current node.

    def insert(self, key, priority=None):
        # Public method to insert a key with an optional priority.
        self.root = self._insert(self.root, key, priority)

    def _delete(self, node, key):
        # Recursive helper function to delete a node with the given key.
        if node is None:
            # Key not found; nothing to delete.
            return None

        if key < node.key:
            # Continue searching in the left subtree.
            node.left = self._delete(node.left, key)
        elif key > node.key:
            # Continue searching in the right subtree.
            node.right = self._delete(node.right, key)
        else:
            # Node to be deleted found.
            if node.left is None:
                # Replace node with its right child.
                return node.right
            elif node.right is None:
                # Replace node with its left child.
                return node.left
            else:
                # Both children exist; rotate to maintain heap property.
                if node.left.priority > node.right.priority:
                    node = self._rotate_right(node)
                    node.right = self._delete(node.right, key)
                else:
                    node = self._rotate_left(node)
                    node.left = self._delete(node.left, key)

        return node  # Return the (possibly updated) current node.

    def delete(self, key):
        # Public method to delete a key from the Treap.
        self.root = self._delete(self.root, key)

    def _search(self, node, key):
        # Recursive helper function to search for a key.
        if node is None or node.key == key:
            # Key found or reached the end of the path.
            return node

        if key < node.key:
            # Search in the left subtree.
            return self._search(node.left, key)
        else:
            # Search in the right subtree.
            return self._search(node.right, key)

    def search(self, key):
        # Public method to search for a key.
        # Returns True if found, False otherwise.
        node = self._search(self.root, key)
        return node is not None

    def _inorder(self, node):
        # In-order traversal to display the Treap.
        if node:
            self._inorder(node.left)
            print(f"Key: {node.key}, Priority: {node.priority}")
            self._inorder(node.right)

    def display(self):
        # Public method to display the Treap.
        print("Treap in-order traversal:")
        self._inorder(self.root)
        print()  # New line for clarity.

# Example usage
if __name__ == "__main__":
    treap = Treap()
    # Insert nodes into the Treap.
    treap.insert(10)
    treap.insert(20)
    treap.insert(15)
    treap.insert(5)
    treap.insert(30)

    # Display the Treap structure.
    treap.display()

    # Search for keys in the Treap.
    print("Search 15:", treap.search(15))
    print("Search 100:", treap.search(100))

    # Delete a key from the Treap.
    treap.delete(20)
    print("After deleting 20:")
    treap.display()
import random

class TreapNode:
    # Class implementation continues here...