import heapq

class EdgeWeight:
    def __init__(self, from_vertex, to_vertex, weight):
        self.from_vertex = from_vertex
        self.to_vertex = to_vertex
        self.weight = weight

    def __lt__(self, other):
        return self.weight < other.weight  # Allows EdgeWeight objects to be compared by weight for the min-heap

class VertexSetCollection:
    def __init__(self, adjacency_list):
        # Initialize each vertex as its own set (disjoint set)
        self.parent = {}
        for vertex in adjacency_list:
            self.parent[vertex] = vertex

    def get_set(self, vertex):
        # Find with path compression
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.get_set(self.parent[vertex])
        return self.parent[vertex]

    def merge(self, set1, set2):
        # Union operation to merge two sets by setting one as the parent of the other
        root1 = self.get_set(set1)
        root2 = self.get_set(set2)
        if root1 != root2:
            self.parent[root2] = root1  # Merge root2 into root1 set

class Graph:
    def __init__(self):
        self.adjacency_list = {}  # Dictionary to store adjacency list
        self.edge_weights = {}    # Dictionary to store edges with weights

    def add_vertex(self, vertex):
        self.adjacency_list[vertex] = []

    def add_edge(self, from_vertex, to_vertex, weight):
        self.adjacency_list[from_vertex].append(to_vertex)
        self.adjacency_list[to_vertex].append(from_vertex)
        self.edge_weights[(from_vertex, to_vertex)] = weight
        self.edge_weights[(to_vertex, from_vertex)] = weight  # For undirected graphs

def minimum_spanning_tree(graph):
    # Create a list of all edges as EdgeWeight objects
    edge_list = []
    for edge in graph.edge_weights:
        edge_weight = EdgeWeight(edge[0], edge[1], graph.edge_weights[edge])
        edge_list.append(edge_weight)

    # Convert edge_list into a priority queue (min-heap)
    heapq.heapify(edge_list)

    # Initialize the collection of vertex sets for the union-find structure
    vertex_sets = VertexSetCollection(graph.adjacency_list)

    result_list = []  # This will store the edges of the MST

    # Process edges until we have |V| - 1 edges in the MST or no edges left
    while len(vertex_sets.parent) > 1 and len(edge_list) > 0:
        # Remove the edge with the minimum weight from edge_list
        next_edge = heapq.heappop(edge_list)

        # Find the sets containing the vertices of the next edge
        set1 = vertex_sets.get_set(next_edge.from_vertex)
        set2 = vertex_sets.get_set(next_edge.to_vertex)

        # If the vertices are in different sets, the edge does not create a cycle
        if set1 != set2:
            # Add next_edge to the MST result list
            result_list.append(next_edge)
            # Merge the two sets to ensure connectivity
            vertex_sets.merge(set1, set2)

    return [(edge.from_vertex.label, edge.to_vertex.label, edge.weight) for edge in result_list]


# Example usage
class Vertex:
    def __init__(self, label):
        self.label = label

vertex_A = Vertex("A")
vertex_B = Vertex("B")
vertex_C = Vertex("C")
vertex_D = Vertex("D")
vertex_E = Vertex("E")

# Create a graph and add vertices
g = Graph()
for vertex in [vertex_A, vertex_B, vertex_C, vertex_D, vertex_E]:
    g.add_vertex(vertex)

# Add edges with weights
g.add_edge(vertex_A, vertex_B, 1)
g.add_edge(vertex_A, vertex_C, 3)
g.add_edge(vertex_B, vertex_C, 3)
g.add_edge(vertex_B, vertex_D, 6)
g.add_edge(vertex_C, vertex_D, 4)
g.add_edge(vertex_C, vertex_E, 2)
g.add_edge(vertex_D, vertex_E, 5)

# Find the minimum spanning tree using Kruskal's algorithm
mst_edges = minimum_spanning_tree(g)
print("Edges in the Minimum Spanning Tree:")
for from_vertex, to_vertex, weight in mst_edges:
    print(f"{from_vertex} - {to_vertex} (Weight: {weight})")

"""
Explanation:
The `minimum_spanning_tree` function uses Kruskal's algorithm to find the minimum spanning tree (MST) of a weighted, undirected graph.
It returns a list of edges in the MST, which connects all vertices with the minimum possible total edge weight.

Steps:
1. **Edge List Initialization**: Create a list of all edges in the graph, wrapped in `EdgeWeight` objects (which store from_vertex, to_vertex, and weight).
   - We use the `EdgeWeight` class to store edges and implement the `__lt__` method so that `heapq` can sort edges by weight.

2. **Priority Queue (Min-Heap)**: Convert `edge_list` into a min-heap using `heapq.heapify`, so we can efficiently retrieve the smallest edge each time.

3. **Union-Find Setup**: Initialize `vertex_sets` (an instance of `VertexSetCollection`) for managing disjoint sets of vertices.
   - Each vertex starts in its own set. The `merge` and `get_set` methods help track connected components and avoid cycles.

4. **Building the MST**:
   - Pop the smallest edge from `edge_list`.
   - If the vertices of the edge belong to different sets (checked using `get_set`), add the edge to `result_list`.
   - Merge the sets of the vertices to ensure they're connected.

5. **Return MST**: The function returns a list of edges in the MST, with each edge represented as (from_vertex.label, to_vertex.label, weight).

Example:
For a sample graph, the output shows the edges that form the MST and their weights, connecting all vertices with the minimum total weight.
"""
