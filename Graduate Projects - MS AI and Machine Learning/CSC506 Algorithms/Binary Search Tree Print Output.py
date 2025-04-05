class Node:
    def __init__(self, key):
        # Initialize a new node with the given key
        self.key = key
        self.left = None  # Left child of the node
        self.right = None  # Right child of the node

class BinarySearchTree:
    def __init__(self):
        # Initialize an empty binary search tree
        self.root = None  # Root of the tree

    def insert(self, key):
        # Insert a new key into the binary search tree
        if self.root is None:  # If the tree is empty
            self.root = Node(key)  # Create a new node as the root
        else:
            self._insert_recursive(self.root, key)  # Recursively insert the key

    def _insert_recursive(self, current_node, key):
        # Recursive helper function for inserting a key
        if key < current_node.key:  # If the key is less than the current node's key
            if current_node.left is None:  # If there is no left child
                current_node.left = Node(key)  # Insert the key as the left child
            else:
                self._insert_recursive(current_node.left, key)  # Recur on the left subtree
        elif key > current_node.key:  # If the key is greater than the current node's key
            if current_node.right is None:  # If there is no right child
                current_node.right = Node(key)  # Insert the key as the right child
            else:
                self._insert_recursive(current_node.right, key)  # Recur on the right subtree

    def search(self, key):
        # Search for a key in the binary search tree
        return self._search_recursive(self.root, key)  # Call the recursive search function

    def _search_recursive(self, current_node, key):
        # Recursive helper function for searching a key
        if current_node is None or current_node.key == key:  # If the node is None or the key matches
            return current_node  # Return the node (or None if not found)
        if key < current_node.key:  # If the key is less than the current node's key
            return self._search_recursive(current_node.left, key)  # Recur on the left subtree
        return self._search_recursive(current_node.right, key)  # Otherwise, recur on the right subtree

    def remove(self, key):
        # Remove a node with a given key from the binary search tree
        self.root = self._remove_recursive(self.root, key)  # Call the recursive remove function

    def _remove_recursive(self, current_node, key):
        # Recursive helper function for removing a node
        if current_node is None:  # If the current node is None, key not found
            return current_node

        if key < current_node.key:  # If the key is less than the current node's key
            current_node.left = self._remove_recursive(current_node.left, key)  # Recur on the left subtree
        elif key > current_node.key:  # If the key is greater than the current node's key
            current_node.right = self._remove_recursive(current_node.right, key)  # Recur on the right subtree
        else:
            # Key matches current node's key, this is the node to be deleted

            # Case 1: No children (leaf node)
            if current_node.left is None and current_node.right is None:
                return None  # Remove the node by returning None

            # Case 2: One child
            elif current_node.left is None:  # Node has only right child
                return current_node.right  # Return the right child to bypass current node
            elif current_node.right is None:  # Node has only left child
                return current_node.left  # Return the left child to bypass current node

            # Case 3: Two children
            # Find the in-order successor (smallest in the right subtree)
            successor = self._find_min(current_node.right)
            current_node.key = successor.key  # Replace current node's key with successor's key
            # Remove the successor node from the right subtree
            current_node.right = self._remove_recursive(current_node.right, successor.key)

        return current_node  # Return the current node to link back in recursion

    def _find_min(self, current_node):
        # Find the minimum value node in the subtree
        while current_node.left is not None:  # Traverse the leftmost path
            current_node = current_node.left  # Go to the left child
        return current_node  # Return the leftmost (smallest) node

    def inorder_traversal(self):
        # Perform in-order traversal of the tree and return the keys in sorted order
        result = []  # Initialize an empty list to store the result
        self._inorder_recursive(self.root, result)  # Call the recursive helper function
        return result  # Return the list of keys in sorted order

    def _inorder_recursive(self, current_node, result):
        # Recursive helper function for in-order traversal
        if current_node is not None:
            self._inorder_recursive(current_node.left, result)  # Recur on the left subtree
            result.append(current_node.key)  # Visit the current node and add its key to the result
            self._inorder_recursive(current_node.right, result)  # Recur on the right subtree

    def print_tree(self):
        # Print the tree structure in a readable format
        self._print_recursive(self.root, 0)  # Call the recursive print function starting at the root

    def _print_recursive(self, current_node, level):
        # Recursive function to print the tree structure
        if current_node is not None:
            self._print_recursive(current_node.right, level + 1)  # Print right subtree
            print('    ' * level + f'-> {current_node.key}')  # Print the current node with indentation
            self._print_recursive(current_node.left, level + 1)  # Print left subtree


# Soliciting input from the user
bst = BinarySearchTree()  # Create an instance of the binary search tree

# Ask the user to input values
print("Enter values to insert into the Binary Search Tree (comma-separated):")
values = input().split(',')  # Take comma-separated input from the user
values = [int(v.strip()) for v in values]  # Convert the input into a list of integers

# Insert the values into the binary search tree
for value in values:
    bst.insert(value)

# Print the tree structure
print("\nBinary Search Tree structure:")
bst.print_tree()

# Optionally, display the sorted order of the tree using in-order traversal
print("\nIn-order traversal (sorted order):")
print(bst.inorder_traversal())

"""
Explanation of the Code:

1. Node Class:
   - The Node class represents a single node in the binary search tree.
   - Each node contains a 'key' value and two pointers: 'left' and 'right' to its children.
   - Initially, both 'left' and 'right' are set to None when the node is created.

2. BinarySearchTree Class:
   - The BinarySearchTree class contains the entire tree structure.
   - It has an attribute 'root' which holds the root node of the tree. Initially, the tree is empty, so 'root' is set to None.

3. Insert Function:
   - The 'insert' function is used to add a new node with the specified key into the tree.
   - If the tree is empty, the new node becomes the root.
   - If the tree is not empty, the function recursively traverses the tree to find the correct position (based on comparisons) and adds the new node as either a left or right child.

4. Search Function:
   - The 'search' function checks if a node with a given key exists in the tree.
   - It recursively traverses the tree, comparing the key with the current node's 

"""