class Vertex:
    # Constructor to initialize the vertex with a label
    def __init__(self, label):
        self.label = label  # Assign the label to the vertex

class Graph:
    # Constructor to initialize the graph with an empty adjacency list
    def __init__(self):
        self.adjacency_list = {}  # Dictionary to store vertices and their connections

    # Method to add a vertex to the graph
    def add_vertex(self, vertex):
        if vertex.label not in self.adjacency_list:  # Check if the vertex label already exists
            self.adjacency_list[vertex.label] = []  # Add the vertex label with an empty list of edges

# Program to create and populate a Graph object
g = Graph()  # Instantiate a Graph object
vertex_a = Vertex("New York")  # Create a vertex labeled "New York"
vertex_b = Vertex("Tokyo")     # Create a vertex labeled "Tokyo"
vertex_c = Vertex("London")    # Create a vertex labeled "London"

g.add_vertex(vertex_a)  # Add vertex "New York" to the graph
g.add_vertex(vertex_b)  # Add vertex "Tokyo" to the graph
g.add_vertex(vertex_c)  # Add vertex "London" to the graph

# Print the adjacency list to verify the structure of the graph
print(g.adjacency_list)  # Output should display each vertex with its empty edges list

"""
Explanation:
1. The `Vertex` class is used to create vertex objects, each having a `label` property to identify the vertex.
2. The `Graph` class manages a collection of vertices and their edges using an adjacency list (a dictionary).
   - `add_vertex`: This method checks if a vertex label is already present in the `adjacency_list`. If not, it adds the label as a key and assigns it an empty list, which will later store connected vertices (edges).
3. In the main program:
   - A `Graph` instance `g` is created.
   - Three `Vertex` objects, labeled "New York," "Tokyo," and "London," are instantiated.
   - Each vertex is added to the graph using `add_vertex`, so the graph's adjacency list now contains entries for these vertices.
4. The `print(g.adjacency_list)` statement outputs the adjacency list, showing the vertices as dictionary keys with empty lists, indicating that no edges have been added yet.

Expected Output:
{'New York': [], 'Tokyo': [], 'London': []}
"""
