class Node:
    def __init__(self, key):
        # Initialize a new node with the given key
        self.key = key
        self.left = None  # Left child of the node
        self.right = None  # Right child of the node
        self.height = 1  # Height of the node for balancing purposes

class AVLTree:
    def __init__(self):
        # Initialize an empty AVL tree
        self.root = None

    def insert(self, key):
        # Insert a new key into the AVL tree and return the new root
        if self.root is None:
            self.root = Node(key)
        else:
            self.root = self._insert_recursive(self.root, key)

    def _insert_recursive(self, current_node, key):
        # Recursive helper function for inserting a key
        if current_node is None:
            return Node(key)

        if key < current_node.key:
            current_node.left = self._insert_recursive(current_node.left, key)
        elif key > current_node.key:
            current_node.right = self._insert_recursive(current_node.right, key)
        else:
            return current_node  # Duplicate keys are not allowed in an AVL tree

        # Update the height of the current node
        current_node.height = 1 + max(self._get_height(current_node.left), self._get_height(current_node.right))

        # Get the balance factor and rebalance the tree if needed
        balance = self._get_balance(current_node)

        # Left Left Case
        if balance > 1 and key < current_node.left.key:
            return self._right_rotate(current_node)

        # Right Right Case
        if balance < -1 and key > current_node.right.key:
            return self._left_rotate(current_node)

        # Left Right Case
        if balance > 1 and key > current_node.left.key:
            current_node.left = self._left_rotate(current_node.left)
            return self._right_rotate(current_node)

        # Right Left Case
        if balance < -1 and key < current_node.right.key:
            current_node.right = self._right_rotate(current_node.right)
            return self._left_rotate(current_node)

        return current_node

    def remove(self, key):
        # Remove a node with a given key from the AVL tree
        if self.root is None:
            return None
        else:
            self.root = self._remove_recursive(self.root, key)

    def _remove_recursive(self, current_node, key):
        # Recursive helper function for removing a node
        if current_node is None:
            return current_node

        if key < current_node.key:
            current_node.left = self._remove_recursive(current_node.left, key)
        elif key > current_node.key:
            current_node.right = self._remove_recursive(current_node.right, key)
        else:
            # Node to be deleted found

            # Case 1: No children (leaf node)
            if current_node.left is None and current_node.right is None:
                return None

            # Case 2: One child
            elif current_node.left is None:
                return current_node.right
            elif current_node.right is None:
                return current_node.left

            # Case 3: Two children
            successor = self._find_min(current_node.right)
            current_node.key = successor.key
            current_node.right = self._remove_recursive(current_node.right, successor.key)

        # Update the height of the current node
        current_node.height = 1 + max(self._get_height(current_node.left), self._get_height(current_node.right))

        # Get the balance factor and rebalance the tree if needed
        balance = self._get_balance(current_node)

        # Left Left Case
        if balance > 1 and self._get_balance(current_node.left) >= 0:
            return self._right_rotate(current_node)

        # Left Right Case
        if balance > 1 and self._get_balance(current_node.left) < 0:
            current_node.left = self._left_rotate(current_node.left)
            return self._right_rotate(current_node)

        # Right Right Case
        if balance < -1 and self._get_balance(current_node.right) <= 0:
            return self._left_rotate(current_node)

        # Right Left Case
        if balance < -1 and self._get_balance(current_node.right) > 0:
            current_node.right = self._right_rotate(current_node.right)
            return self._left_rotate(current_node)

        return current_node

    def _left_rotate(self, z):
        # Perform a left rotation on the node z
        y = z.right
        T2 = y.left

        # Perform rotation
        y.left = z
        z.right = T2

        # Update heights
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

        return y

    def _right_rotate(self, z):
        # Perform a right rotation on the node z
        y = z.left
        T3 = y.right

        # Perform rotation
        y.right = z
        z.left = T3

        # Update heights
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

        return y

    def _get_height(self, node):
        # Return the height of the node
        if node is None:
            return 0
        return node.height

    def _get_balance(self, node):
        # Return the balance factor of the node
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _find_min(self, current_node):
        # Find the minimum value node in the subtree
        while current_node.left is not None:
            current_node = current_node.left
        return current_node

    def inorder_traversal(self):
        # Perform in-order traversal of the tree and return the keys in sorted order
        result = []  # Initialize an empty list to store the result
        self._inorder_recursive(self.root, result)  # Call the recursive helper function
        return result  # Return the list of keys in sorted order

    def _inorder_recursive(self, current_node, result):
        # Recursive helper function for in-order traversal
        if current_node is not None:
            self._inorder_recursive(current_node.left, result)
            result.append(current_node.key)
            self._inorder_recursive(current_node.right, result)

    def print_tree(self):
        # Print the tree structure in a readable format
        self._print_recursive(self.root, 0)

    def _print_recursive(self, current_node, level):
        # Recursive function to print the tree structure
        if current_node is not None:
            self._print_recursive(current_node.right, level + 1)  # Print right subtree
            print('    ' * level + f'-> {current_node.key}')  # Print the current node with indentation
            self._print_recursive(current_node.left, level + 1)  # Print left subtree


# Insert 64 random numbers into the AVL tree
avl_tree = AVLTree()  # Create an instance of the AVL tree

# 64 random numbers for input
values = [976, 521, 146, 74, 136, 570, 604, 692, 543, 195, 514, 686, 787, 58, 625, 299, 672, 87, 388, 101, 968, 532, 868, 403, 85, 782, 856, 177, 627, 847, 487, 60, 985, 447, 602, 2, 781, 411, 511, 880, 157, 790, 508, 489, 933, 485, 255, 75, 436, 502, 271, 864, 143, 564, 720, 61, 228, 345, 305, 600, 426, 309, 145, 90]

# Insert the values into the AVL tree
for value in values:
    avl_tree.insert(value)

# Print the tree structure
print("\nAVL Tree structure:")
avl_tree.print_tree()

# Optionally, display the sorted order of the tree using in-order traversal
print("\nIn-order traversal (sorted order):")
print(avl_tree.inorder_traversal())

"""
Explanation of the Code:

1. Node Class:
   - The Node class represents a single node in the AVL tree.
   - Each node contains a 'key' value, two pointers ('left' and 'right'), and a 'height' attribute to help balance the tree.

2. AVLTree Class:
   - The AVLTree class contains the entire tree structure and ensures the tree remains balanced through rotations.
   - After each insertion or removal, the tree checks the balance factor and rotates nodes if necessary.

3. Insert Function:
   - The 'insert' function adds a new node to the tree.
   - After each insertion, the tree recalculates the balance factor and performs rotations if the tree becomes unbalanced.

4. Remove Function:
   - The 'remove' function deletes a node from the tree.
   - The balance factor is checked after each deletion and rebalanced if necessary.

5. In-order Traversal:

