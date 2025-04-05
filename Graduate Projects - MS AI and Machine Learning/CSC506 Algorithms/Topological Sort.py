class Vertex:
    def __init__(self, label):
        self.label = label  # Assign label to the vertex

class Graph:
    def __init__(self):
        self.adjacency_list = {}  # Dictionary to store adjacency list
        self.edge_weights = {}    # Dictionary to store edge weights (edges)

    def add_vertex(self, vertex):
        self.adjacency_list[vertex] = []  # Initialize an empty adjacency list for the vertex

    def add_edge(self, from_vertex, to_vertex, weight=1):
        self.adjacency_list[from_vertex].append(to_vertex)  # Add directed edge from from_vertex to to_vertex
        self.edge_weights[(from_vertex, to_vertex)] = weight  # Set the weight of the edge (default is 1)

def get_incoming_edge_count(edges, vertex):
    """Helper function to count incoming edges for a vertex."""
    count = 0
    for (from_vertex, to_vertex) in edges:  # Iterate over all edges
        if to_vertex == vertex:             # If edge points to the target vertex
            count += 1                      # Increment the count
    return count

def topological_sort(graph):
    result_list = []      # List to store the final topological order
    no_incoming = []      # List to store vertices with no incoming edges

    # Initialize list of vertices with no incoming edges
    for vertex in graph.adjacency_list.keys():  # Iterate over all vertices
        if get_incoming_edge_count(graph.edge_weights.keys(), vertex) == 0:
            no_incoming.append(vertex)  # Add vertices with no incoming edges to no_incoming

    # Set of all remaining edges in the graph, used to manage dependencies
    remaining_edges = set(graph.edge_weights.keys())  # Initialize set of all edges for quick removal

    # Process each vertex in no_incoming until all vertices are sorted
    while len(no_incoming) != 0:
        # Select the next vertex to add to the topological order
        current_vertex = no_incoming.pop()  # Remove vertex from no_incoming
        result_list.append(current_vertex)  # Add vertex to the result list

        outgoing_edges = []  # List to keep track of outgoing edges from current_vertex

        # Remove all outgoing edges from current_vertex in remaining_edges
        for to_vertex in graph.adjacency_list[current_vertex]:  # Check each neighbor
            outgoing_edge = (current_vertex, to_vertex)
            if outgoing_edge in remaining_edges:
                outgoing_edges.append(outgoing_edge)   # Track outgoing edges
                remaining_edges.remove(outgoing_edge)  # Remove edge from remaining_edges

        # Check if removing edges has created any new vertices with no incoming edges
        for (from_vertex, to_vertex) in outgoing_edges:
            in_count = get_incoming_edge_count(remaining_edges, to_vertex)  # Count remaining incoming edges
            if in_count == 0:  # If no incoming edges left
                no_incoming.append(to_vertex)  # Add vertex to no_incoming

    # Return the topological sort result as a list of vertex labels
    return [vertex.label for vertex in result_list]


# Example usage
# Create vertices
vertex_A = Vertex("A")
vertex_B = Vertex("B")
vertex_C = Vertex("C")
vertex_D = Vertex("D")
vertex_E = Vertex("E")
vertex_F = Vertex("F")

# Create a graph and add vertices
g = Graph()
for vertex in [vertex_A, vertex_B, vertex_C, vertex_D, vertex_E, vertex_F]:
    g.add_vertex(vertex)

# Add directed edges to form a DAG
g.add_edge(vertex_A, vertex_C)
g.add_edge(vertex_B, vertex_C)
g.add_edge(vertex_B, vertex_D)
g.add_edge(vertex_C, vertex_E)
g.add_edge(vertex_D, vertex_F)
g.add_edge(vertex_E, vertex_F)

# Perform topological sort
sorted_order = topological_sort(g)
print("Topological Sort Order:", sorted_order)

"""
Explanation:
The topological sort function performs a linear ordering of vertices in a directed acyclic graph (DAG)
such that for every directed edge X → Y, vertex X comes before vertex Y in the ordering.

Steps:
1. **Initialize no_incoming**: Identify vertices with no incoming edges and add them to `no_incoming`.
2. **Process each vertex in no_incoming**:
   - Pop a vertex from `no_incoming` and add it to `result_list` as it has no dependencies.
   - Remove all outgoing edges from this vertex in `remaining_edges`.
3. **Identify new vertices with no incoming edges**:
   - After removing outgoing edges, check if any connected vertices have no remaining incoming edges.
   - If so, add those vertices to `no_incoming` for further processing.
4. **Repeat** until all vertices are processed.

Return:
- A list of vertex labels in topological order.

Assumptions:
- `Graph` and `Vertex` classes are defined with an adjacency list and edge weights.

Sample Output:
The output for the example DAG will display the vertices in a topological order, such as:
    "Topological Sort Order: ['B', 'A', 'C', 'D', 'E', 'F']"
"""
