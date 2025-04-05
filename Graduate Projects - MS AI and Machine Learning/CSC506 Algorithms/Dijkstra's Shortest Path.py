class Vertex:
    def __init__(self, label):
        self.label = label  # Name or identifier of the vertex
        self.distance = float('inf')  # Distance from the start vertex, initialized to infinity
        self.pred_vertex = None  # Predecessor vertex, used to trace the path

class Graph:
    def __init__(self):
        self.adjacency_list = {}  # Dictionary to store adjacency list
        self.edge_weights = {}  # Dictionary to store edge weights

    def add_vertex(self, vertex):
        self.adjacency_list[vertex] = []  # Initialize the adjacency list for the vertex

    def add_edge(self, from_vertex, to_vertex, weight):
        self.adjacency_list[from_vertex].append(to_vertex)  # Add the directed edge to adjacency list
        self.edge_weights[(from_vertex, to_vertex)] = weight  # Set the weight of the edge

def dijkstra_shortest_path(g, start_vertex, end_vertex):
    # Initialize all vertices with infinite distance and no predecessor
    unvisited_queue = []  # List to hold unvisited vertices
    for vertex in g.adjacency_list:
        vertex.distance = float('inf')  # Set initial distance to infinity for all vertices
        vertex.pred_vertex = None       # Set predecessor to None to mark no prior connections
        unvisited_queue.append(vertex)  # Add vertex to the unvisited queue

    # Set the starting vertex's distance to 0 to begin the path from here
    start_vertex.distance = 0

    # Main loop of Dijkstra's algorithm to calculate shortest path distances
    while len(unvisited_queue) > 0:

        # Find the vertex in the unvisited queue with the smallest distance
        smallest_index = 0
        for i in range(1, len(unvisited_queue)):
            if unvisited_queue[i].distance < unvisited_queue[smallest_index].distance:
                smallest_index = i
        # Remove and process the vertex with the smallest distance
        current_vertex = unvisited_queue.pop(smallest_index)

        # Check all adjacent vertices to find potential shorter paths
        for adj_vertex in g.adjacency_list[current_vertex]:
            # Retrieve the weight of the edge connecting current_vertex to adj_vertex
            edge_weight = g.edge_weights[(current_vertex, adj_vertex)]
            # Calculate alternative path distance from start_vertex through current_vertex to adj_vertex
            alternative_path_distance = current_vertex.distance + edge_weight

            # Update distance and predecessor if a shorter path to adj_vertex is found
            if alternative_path_distance < adj_vertex.distance:
                adj_vertex.distance = alternative_path_distance  # Update with the new shorter distance
                adj_vertex.pred_vertex = current_vertex         # Set current_vertex as predecessor

    # Construct the shortest path from start_vertex to end_vertex
    path = ''
    current_vertex = end_vertex
    # Traverse back from end_vertex to start_vertex using predecessor pointers
    while current_vertex is not start_vertex:
        if current_vertex is None:  # Return message if no path exists
            return "No path found"
        # Build the path string in reverse order by adding current vertex's label
        path = ' -> ' + str(current_vertex.label) + path
        current_vertex = current_vertex.pred_vertex  # Move to the predecessor
    # Add the starting vertex's label to complete the path
    path = start_vertex.label + path  

    return path  # Return the shortest path as a formatted string

"""
    This function implements Dijkstra's algorithm to find the shortest path between a start and an end vertex in a weighted graph. 

    1. First, it initializes each vertex's distance to infinity and its predecessor to None, except for the start vertex, whose distance is set to 0.
    2. It then iteratively finds the vertex with the smallest distance in the unvisited set, removes it, and calculates potential shorter path distances to each of its adjacent vertices.
    3. If a shorter path to an adjacent vertex is found, it updates the distance and predecessor of that vertex.
    4. After the shortest path distances are calculated, the function constructs the path from the end vertex back to the start vertex by following predecessor pointers. 
    5. The path is formatted as a string and returned. If no path exists, it returns "No path found."

    The function assumes:
    - `g.adjacency_list` contains a dictionary with vertices as keys and lists of adjacent vertices as values.
    - `g.edge_weights` stores edge weights as a dictionary with (vertex1, vertex2) tuples as keys.
"""

# Example usage
# Create vertices
vA = Vertex("A")
vB = Vertex("B")
vC = Vertex("C")
vD = Vertex("D")
vE = Vertex("E")

# Create a graph and add vertices
graph = Graph()
for v in [vA, vB, vC, vD, vE]:
    graph.add_vertex(v)

# Add edges with weights
graph.add_edge(vA, vB, 4)
graph.add_edge(vA, vC, 1)
graph.add_edge(vC, vB, 2)
graph.add_edge(vB, vD, 5)
graph.add_edge(vC, vD, 8)
graph.add_edge(vD, vE, 3)
graph.add_edge(vC, vE, 10)

# Calculate and print the shortest path from vertex A to vertex E
shortest_path = dijkstra_shortest_path(graph, vA, vE)
print("Shortest path from A to E:", shortest_path)
