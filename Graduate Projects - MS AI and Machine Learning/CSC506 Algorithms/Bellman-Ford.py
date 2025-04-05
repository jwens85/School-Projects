def bellman_ford(graph, start_vertex):
  # Initialize all vertex distances to infinity and predecessors to None.
  for current_vertex in graph.adjacency_list:
      current_vertex.distance = float('inf')  # Set distance to infinity for all vertices initially
      current_vertex.pred_vertex = None       # Set predecessor to None to mark no prior connections

  # Set the start vertex's distance to 0 (distance to itself)
  start_vertex.distance = 0                

  # Main loop executed |V|-1 times to guarantee minimum distances.
  # This loop updates distances based on the edges in the graph
  for i in range(len(graph.adjacency_list) - 1):  # Run |V| - 1 times
      for current_vertex in graph.adjacency_list:
          # For each neighbor (adjacent vertex) of current_vertex
          for adj_vertex in graph.adjacency_list[current_vertex]:
              edge_weight = graph.edge_weights[(current_vertex, adj_vertex)]  # Get the weight of the edge
              # Calculate the distance of the alternative path through current_vertex
              alternative_path_distance = current_vertex.distance + edge_weight

              # If a shorter path to adj_vertex is found, update its distance and predecessor
              if alternative_path_distance < adj_vertex.distance:
                  adj_vertex.distance = alternative_path_distance  # Update distance with the shorter path
                  adj_vertex.pred_vertex = current_vertex          # Set current_vertex as predecessor

  # Check for negative edge weight cycles by trying one more relaxation
  for current_vertex in graph.adjacency_list:
      for adj_vertex in graph.adjacency_list[current_vertex]:
          edge_weight = graph.edge_weights[(current_vertex, adj_vertex)]
          alternative_path_distance = current_vertex.distance + edge_weight

          # If a shorter path is found in this check, a negative weight cycle exists
          if alternative_path_distance < adj_vertex.distance:
              return False  # Negative weight cycle detected

  return True  # No negative weight cycle detected, distances are valid

"""
The `bellman_ford` function implements the Bellman-Ford algorithm to find the shortest path from a given start vertex to all other vertices in a weighted graph, even when negative weights are present.

1. **Initialization**:
   - Each vertex's distance is set to infinity (`float('inf')`), and predecessors are set to `None`.
   - The start vertex has a distance of 0 from itself.

2. **Main Relaxation Loop**:
   - The algorithm iterates |V|-1 times (where |V| is the number of vertices). This ensures that the shortest path is found for each vertex, as each iteration allows the algorithm to relax the edges.
   - For each vertex and its adjacent vertices, it calculates the "alternative path distance" by adding the current vertex's distance to the edge weight.
   - If the alternative path is shorter than the current distance to the adjacent vertex, the distance and predecessor for that vertex are updated.

3. **Negative Cycle Check**:
   - After |V|-1 iterations, the function performs an additional pass to check for negative weight cycles.
   - If any distance can still be updated in this pass, a negative weight cycle exists, and the function returns `False`.
   - If no further updates are found, the function returns `True`, indicating no negative weight cycles exist.

The function assumes:
- `graph.adjacency_list` contains a dictionary of vertices with lists of adjacent vertices.
- `graph.edge_weights` stores edge weights in a dictionary with (vertex1, vertex2) tuples as keys.

Return:
- `True` if no negative weight cycles exist.
- `False` if a negative weight cycle is detected.
"""
