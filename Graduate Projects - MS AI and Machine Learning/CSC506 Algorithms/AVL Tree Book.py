class Node:
    # Constructor with a key parameter creates the Node object.
    def __init__(self, key):
        self.key = key
        self.parent = None
        self.left = None
        self.right = None
        self.height = 0
        
    # Calculate the current nodes' balance factor,
    # defined as height(left subtree) - height(right subtree)
    def get_balance(self):
        # Get current height of left subtree, or -1 if None
        left_height = -1
        if self.left is not None:
            left_height = self.left.height
            
        # Get current height of right subtree, or -1 if None
        right_height = -1
        if self.right is not None:
            right_height = self.right.height
            
        # Calculate the balance factor.
        return left_height - right_height

    # Recalculate the current height of the subtree rooted at
    # the node, usually called after a subtree has been 
    # modified.
    def update_height(self):
        # Get current height of left subtree, or -1 if None
        left_height = -1
        if self.left is not None:
            left_height = self.left.height
            
        # Get current height of right subtree, or -1 if None
        right_height = -1
        if self.right is not None:
            right_height = self.right.height

        # Assign self.height with calculated node height.
        self.height = max(left_height, right_height) + 1
# Assign either the left or right data member with a new
# child. The parameter which_child is expected to be the
# string "left" or the string "right". Returns True if
# the new child is successfully assigned to this node, False
# otherwise.
def set_child(self, which_child, child):
    # Ensure which_child is properly assigned.
    if which_child != "left" and which_child != "right":
        return False

    # Assign the left or right data member.
    if which_child == "left":
        self.left = child
    else:
        self.right = child

    # Assign the parent data member of the new child,
    # if the child is not None.
    if child is not None:
        child.parent = self

    # Update the node's height, since the subtree's structure
    # may have changed.
    self.update_height()
    return True

# Replace a current child with a new child. Determines if
# the current child is on the left or right, and calls
# set_child() with the new node appropriately.
# Returns True if the new child is assigned, False otherwise.
def replace_child(self, current_child, new_child):
    if self.left is current_child:
        return self.set_child("left", new_child)
    elif self.right is current_child:
        return self.set_child("right", new_child)
      
    # If neither of the above cases applied, then the new child
    # could not be attached to this node.
    return False