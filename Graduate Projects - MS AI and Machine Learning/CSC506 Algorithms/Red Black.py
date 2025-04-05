class Node:
    def __init__(self, key, color="red"):
        # Initialize a new node with the given key and color (default to red)
        self.key = key
        self.color = color  # Red or Black
        self.left = None  # Left child
        self.right = None  # Right child
        self.parent = None  # Parent node


class RedBlackTree:
    def __init__(self):
        # Initialize the Red-Black Tree with a None root
        self.nil = Node(key=None, color="black")  # Sentinel nil node
        self.root = self.nil

    def left_rotate(self, x):
        # Perform a left rotation around node x
        y = x.right
        x.right = y.left

        if y.left != self.nil:
            y.left.parent = x

        y.parent = x.parent

        if x.parent is None:  # x is root
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y

        y.left = x
        x.parent = y

    def right_rotate(self, y):
        # Perform a right rotation around node y
        x = y.left
        y.left = x.right

        if x.right != self.nil:
            x.right.parent = y

        x.parent = y.parent

        if y.parent is None:  # y is root
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x

        x.right = y
        y.parent = x

    def insert(self, key):
        # Insert a new node with the given key
        new_node = Node(key)
        new_node.left = self.nil
        new_node.right = self.nil

        y = None
        x = self.root

        # Find the correct position to insert the new node
        while x != self.nil:
            y = x
            if new_node.key < x.key:
                x = x.left
            else:
                x = x.right

        new_node.parent = y

        if y is None:
            self.root = new_node  # Tree was empty
        elif new_node.key < y.key:
            y.left = new_node
        else:
            y.right = new_node

        new_node.color = "red"
        self.insert_fixup(new_node)

    def insert_fixup(self, z):
        # Fix the red-black tree after insertion to maintain properties
        while z.parent and z.parent.color == "red":
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right  # Uncle of z
                if y.color == "red":
                    # Case 1: Uncle is red
                    z.parent.color = "black"
                    y.color = "black"
                    z.parent.parent.color = "red"
                    z = z.parent.parent
                else:
                    if z == z.parent.right:
                        # Case 2: Uncle is black, and z is a right child
                        z = z.parent
                        self.left_rotate(z)
                    # Case 3: Uncle is black, and z is a left child
                    z.parent.color = "black"
                    z.parent.parent.color = "red"
                    self.right_rotate(z.parent.parent)
            else:
                y = z.parent.parent.left  # Uncle of z
                if y.color == "red":
                    # Case 1: Uncle is red
                    z.parent.color = "black"
                    y.color = "black"
                    z.parent.parent.color = "red"
                    z = z.parent.parent
                else:
                    if z == z.parent.left:
                        # Case 2: Uncle is black, and z is a left child
                        z = z.parent
                        self.right_rotate(z)
                    # Case 3: Uncle is black, and z is a right child
                    z.parent.color = "black"
                    z.parent.parent.color = "red"
                    self.left_rotate(z.parent.parent)

        self.root.color = "black"

    def transplant(self, u, v):
        # Replaces the subtree rooted at u with the subtree rooted at v
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def remove(self, key):
        # Remove a node with the given key
        z = self.search(key)
        if z is None:
            print(f"Key {key} not found!")
            return

        y = z
        y_original_color = y.color
        if z.left == self.nil:
            x = z.right
            self.transplant(z, z.right)
        elif z.right == self.nil:
            x = z.left
            self.transplant(z, z.left)
        else:
            y = self.tree_minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent == z:
                x.parent = y
            else:
                self.transplant(y, y.right)
                y.right = z.right
                y.right.parent = y

            self.transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color

        if y_original_color == "black":
            self.remove_fixup(x)

    def remove_fixup(self, x):
        # Fix the red-black tree after deletion to maintain properties
        while x != self.root and x.color == "black":
            if x == x.parent.left:
                w = x.parent.right  # Sibling of x
                if w.color == "red":
                    # Case 1: Sibling is red
                    w.color = "black"
                    x.parent.color = "red"
                    self.left_rotate(x.parent)
                    w = x.parent.right

                if w.left.color == "black" and w.right.color == "black":
                    # Case 2: Sibling's children are both black
                    w.color = "red"
                    x = x.parent
                else:
                    if w.right.color == "black":
                        # Case 3: Sibling's right child is black
                        w.left.color = "black"
                        w.color = "red"
                        self.right_rotate(w)
                        w = x.parent.right

                    # Case 4: Sibling's right child is red
                    w.color = x.parent.color
                    x.parent.color = "black"
                    w.right.color = "black"
                    self.left_rotate(x.parent)
                    x = self.root
            else:
                w = x.parent.left  # Sibling of x
                if w.color == "red":
                    # Case 1: Sibling is red
                    w.color = "black"
                    x.parent.color = "red"
                    self.right_rotate(x.parent)
                    w = x.parent.left

                if w.right.color == "black" and w.left.color == "black":
                    # Case 2: Sibling's children are both black
                    w.color = "red"
                    x = x.parent
                else:
                    if w.left.color == "black":
                        # Case 3: Sibling's left child is black
                        w.right.color = "black"
                        w.color = "red"
                        self.left_rotate(w)
                        w = x.parent.left

                    # Case 4: Sibling's left child is red
                    w.color = x.parent.color
                    x.parent.color = "black"
                    w.left.color = "black"
                    self.right_rotate(x.parent)
                    x = self.root
        x.color = "black"

    def search(self, key):
        # Search the tree for a node with the given key
        node = self.root
        while node != self.nil and key != node.key:
            if key < node.key:
                node = node.left
            else:
                node = node.right
        return node if node != self.nil else None

    def tree_minimum(self, node):
        # Get the minimum node starting from the given node
        while node.left != self.nil:
            node = node.left
        return node

    def inorder_traversal(self, node=None, result=None):
        # Perform in-order traversal of the tree and return keys in sorted order
        if result is None:
            result = []
        if node is None:
            node = self.root
        if node != self.nil:
            self.inorder_traversal(node.left, result)
            result.append((node.key, node.color))  # Include color in the output
            self.inorder_traversal(node.right, result)
        return result

    def print_tree(self):
        # Print the tree structure
        def print_helper(node, indent, last):
            if node != self.nil:
                print(indent, end="")
                if last:
                    print("R----", end="")
                    indent += "     "
                else:
                    print("L----", end="")


"""
Explanation of the Code:

1. Node Class:
   - The Node class represents a node in the Red-Black Tree.
   - Each node has a key and a color attribute (either "red" or "black").
   - The node also has pointers to its left child, right child, and parent.
   - The tree uses a sentinel node (`nil`), which represents all leaf nodes as black nodes.

2. RedBlackTree Class:
   - This class manages the entire Red-Black Tree structure, including maintaining balance during insertion and deletion.
   - The tree begins with an empty root, which points to the sentinel `nil` node.

3. Left and Right Rotations:
   - **left_rotate(x)**: This performs a left rotation around node `x`, shifting the tree structure to maintain balance.
   - **right_rotate(y)**: This performs a right rotation around node `y`, shifting the tree to maintain balance.
   - Rotations are essential for maintaining the Red-Black Tree's balance property.

4. Insertion and Fix-Up:
   - **insert(key)**: Inserts a node with the given key. Initially, the inserted node is colored red.
   - **insert_fixup(z)**: This method fixes any violations of the Red-Black Tree properties after insertion by checking the colors of the nodes and rotating/re-coloring as necessary.
   - The `fixup` process is crucial because a red node cannot have a red child, and the tree must maintain its balance properties.

5. Transplant Method:
   - **transplant(u, v)**: This method replaces the subtree rooted at `u` with the subtree rooted at `v`. It is used during deletion to replace a node with its successor.

6. Deletion and Fix-Up:
   - **remove(key)**: This removes a node with the specified key from the Red-Black Tree, ensuring the tree stays balanced afterward.
   - **remove_fixup(x)**: After removing a node, this method fixes any violations of the Red-Black Tree properties. It ensures that the tree's properties are maintained by re-coloring and performing rotations as necessary.
   - The `fixup` is particularly important to handle cases where the node being deleted is black, which can disrupt the number of black nodes along a path.

7. Traversal and Tree Printing:
   - **inorder_traversal()**: Returns the in-order traversal of the Red-Black Tree, which gives the nodes in sorted order. The traversal also includes each node's color.
   - **print_tree()**: This method prints the structure of the Red-Black Tree, visually showing the key and color of each node.

8. Example Usage:
   - In this example, the tree is populated with a series of insertions and deletions, and then the structure and in-order traversal are printed to demonstrate the Red-Black Tree in action.
"""
