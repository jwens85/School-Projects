def get_shortest_path(start_vertex, end_vertex):
  # Initialize an empty path string
  path = ''
  # Start from the end vertex and build the path backward to the start vertex
  current_vertex = end_vertex
  while current_vertex is not start_vertex:  # Continue until reaching the start vertex
      path = ' -> ' + str(current_vertex.label) + path  # Append the current vertex to the path
      current_vertex = current_vertex.pred_vertex  # Move to the predecessor of the current vertex
  path = start_vertex.label + path  # Add the start vertex at the beginning of the path
  return path  # Return the complete path as a string

"""
Explanation:
1. The `get_shortest_path` function is used after running Dijkstra's algorithm to reconstruct the shortest path from `start_vertex` to `end_vertex`.

2. Parameters:
   - `start_vertex`: The starting vertex of the path.
   - `end_vertex`: The destination vertex for which the shortest path is required.

3. Functionality:
   - `path`: An initially empty string that stores the path in reverse order.
   - `current_vertex`: Starts from `end_vertex` and iteratively moves backward through each vertex's `pred_vertex` until it reaches `start_vertex`.
   - Each vertex is added to the path in reverse order (from `end_vertex` to `start_vertex`), and the path string is updated to reflect the correct order (e.g., `A -> B -> C`).

4. End Result:
   - The function returns a string showing the path from `start_vertex` to `end_vertex`, assuming each vertex’s `pred_vertex` attribute has been correctly set by Dijkstra’s algorithm.

This function is essential for visualizing or retrieving the actual sequence of vertices in the shortest path, complementing Dijkstra’s distance calculations.
"""
