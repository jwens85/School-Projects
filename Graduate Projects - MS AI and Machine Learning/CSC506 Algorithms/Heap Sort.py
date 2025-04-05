def heapify(arr, n, i):
    # Initialize the largest element as the root at index i
    largest = i
    # Calculate the left child index of the current node
    left = 2 * i + 1
    # Calculate the right child index of the current node
    right = 2 * i + 2

    # If the left child exists and is greater than the current largest, update largest
    if left < n and arr[largest] < arr[left]:
        largest = left

    # If the right child exists and is greater than the current largest, update largest
    if right < n and arr[largest] < arr[right]:
        largest = right

    # If the largest element is not the root node, swap it with the root
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # Perform the swap

        # Recursively call heapify on the affected subtree to maintain the heap property
        heapify(arr, n, largest)

def heap_sort(arr):
    # Get the number of elements in the array
    n = len(arr)

    # Build a max heap by applying heapify from the last non-leaf node up to the root
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements from the heap one by one to build the sorted array
    for i in range(n - 1, 0, -1):
        # Move the current root (largest) to the end of the unsorted part of the array
        arr[i], arr[0] = arr[0], arr[i]  # Perform the swap
        # Call heapify on the reduced heap to maintain the heap property
        heapify(arr, i, 0)

# Example usage
if __name__ == "__main__":
    # Initialize the array with sample values
    arr = [12, 11, 13, 5, 6, 7]
    # Print the original unsorted array
    print("Original array:", arr)
    # Sort the array using heap sort
    heap_sort(arr)
    # Print the sorted array
    print("Sorted array:", arr)

"""
Summary:
This code implements the Heap Sort algorithm using a max-heap data structure to sort an array in ascending order.

1. **heapify()**:
   - This function ensures that a subtree rooted at index `i` maintains the max-heap property.
   - It checks the left and right children to find the largest value and swaps it with the root if necessary.
   - If a swap occurs, `heapify()` is called recursively on the affected subtree to restore the heap structure.

2. **heap_sort()**:
   - The main sorting function first builds a max-heap by calling `heapify()` on each non-leaf node, starting from the last.
   - It then extracts the maximum element (root of the heap) by swapping it with the last element in the unsorted portion of the array.
   - After each extraction, it reduces the heap size and calls `heapify()` to maintain the max-heap property on the reduced heap.

The example usage sorts the array `[12, 11, 13, 5, 6, 7]` and outputs the sorted array.
"""
