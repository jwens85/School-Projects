from queue import Queue  # Importing Queue for the BFS frontier

class Vertex:
    def __init__(self, label):
        self.label = label       # Label of the vertex
        self.distance = None     # Distance attribute for BFS tracking, initially set to None

# Breadth-first search function
def breadth_first_search(graph, start_vertex):
    discovered_set = set()  # Set to store discovered vertices for O(1) lookups
    frontier_queue = Queue()  # Queue to manage the current frontier in BFS

    start_vertex.distance = 0  # Set the starting vertex's distance to 0
    frontier_queue.put(start_vertex)  # Enqueue the starting vertex
    discovered_set.add(start_vertex)  # Mark the starting vertex as discovered

    traversal_order = []  # List to store the order of visited vertices

    while not frontier_queue.empty():  # While there are vertices to explore
        current_vertex = frontier_queue.get()  # Dequeue the next vertex to explore
        traversal_order.append(current_vertex)  # Record the current vertex in traversal order

        for adjacent_vertex in graph.adjacency_list[current_vertex]:  # Explore adjacent vertices
            if adjacent_vertex not in discovered_set:  # If the adjacent vertex is not discovered
                adjacent_vertex.distance = current_vertex.distance + 1  # Set its distance
                frontier_queue.put(adjacent_vertex)  # Enqueue the adjacent vertex
                discovered_set.add(adjacent_vertex)  # Mark it as discovered

    return traversal_order  # Return the list of vertices in the order they were visited

"""
Explanation:
1. The `Vertex` class initializes each vertex with a label and a distance attribute, which is set to `None` initially.
2. The `breadth_first_search` function performs a breadth-first traversal on the graph.
   - `discovered_set` keeps track of all discovered vertices to prevent re-processing.
   - `frontier_queue` holds the vertices as they are discovered but not yet fully explored.
   - The starting vertex’s `distance` is set to 0, indicating the beginning of the search.
   - The traversal proceeds by visiting each vertex, exploring its neighbors, setting their distances,
     and adding them to the queue if they haven't been visited yet.
3. The function returns `traversal_order`, a list of vertices in the order they were visited.
"""
