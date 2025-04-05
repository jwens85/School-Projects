class Graph:
  # Constructor to initialize the graph with an empty adjacency list and edge weights
  def __init__(self):
      self.adjacency_list = {}  # Dictionary to store vertices and their connected vertices (edges)
      self.edge_weights = {}    # Dictionary to store the weight of each edge

  # Method to add a new vertex to the graph
  def add_vertex(self, new_vertex):
      self.adjacency_list[new_vertex] = []  # Initialize an empty list of edges for the new vertex

  # Method to add a directed edge with an optional weight (default is 1.0)
  def add_directed_edge(self, from_vertex, to_vertex, weight=1.0):
      self.edge_weights[(from_vertex, to_vertex)] = weight  # Store the weight of the directed edge
      self.adjacency_list[from_vertex].append(to_vertex)    # Add the 'to_vertex' to 'from_vertex' adjacency list

  # Method to add an undirected edge between two vertices with an optional weight
  def add_undirected_edge(self, vertex_a, vertex_b, weight=1.0):
      self.add_directed_edge(vertex_a, vertex_b, weight)    # Add edge from vertex_a to vertex_b
      self.add_directed_edge(vertex_b, vertex_a, weight)    # Add edge from vertex_b to vertex_a


"""
Explanation:
1. The `Graph` class includes an adjacency list for representing vertices and edges and a dictionary `edge_weights` to store edge weights.
2. `add_vertex`: Adds a new vertex to the adjacency list with an empty list to hold its connections.
3. `add_directed_edge`: Adds a directed edge from `from_vertex` to `to_vertex` with an optional weight. 
 - It updates `edge_weights` with the weight for this edge.
 - The adjacency list for `from_vertex` is updated to include `to_vertex`.
4. `add_undirected_edge`: Creates a two-way (undirected) connection between `vertex_a` and `vertex_b`.
 - This is achieved by calling `add_directed_edge` in both directions, ensuring both vertices are connected to each other with the specified weight.

This setup supports both directed and undirected graphs, as well as weighted edges.
"""
