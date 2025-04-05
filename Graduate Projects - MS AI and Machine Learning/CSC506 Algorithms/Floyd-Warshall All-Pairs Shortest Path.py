# Define a very large number to represent infinity
INF = float('inf')

def floyd_warshall(graph):
    # Number of vertices in the graph
    vertices = list(graph.keys())
    n = len(vertices)

    # Initialize the distance matrix
    dist = {v: {u: INF for u in vertices} for v in vertices}
    for v in vertices:
        dist[v][v] = 0  # Distance from a vertex to itself is zero

    # Fill in initial distances based on direct edges in the graph
    for v in graph:
        for u, weight in graph[v].items():
            dist[v][u] = weight  # Set the distance for direct edges

    # Floyd-Warshall algorithm to compute shortest paths
    for k in vertices:
        for i in vertices:
            for j in vertices:
                # Update dist[i][j] if a shorter path is found via vertex k
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist  # Return the distance matrix with shortest paths

# Example usage
# Define a graph as an adjacency list with weights
graph = {
    'A': {'B': 3, 'C': 8, 'E': -4},
    'B': {'D': 1, 'E': 7},
    'C': {'B': 4},
    'D': {'A': 2, 'C': -5},
    'E': {'D': 6}
}

# Run the Floyd-Warshall algorithm
shortest_paths = floyd_warshall(graph)

# Display the shortest paths between all pairs of vertices
print("Shortest paths between all pairs of vertices:")
for i in shortest_paths:
    for j in shortest_paths[i]:
        if shortest_paths[i][j] == INF:
            print(f"{i} to {j}: No path")
        else:
            print(f"{i} to {j}: {shortest_paths[i][j]}")

"""
Explanation of Floyd-Warshall Algorithm:

The `floyd_warshall` function computes the shortest paths between all pairs of vertices in a weighted graph.
It returns a distance matrix `dist`, where `dist[i][j]` represents the shortest path distance from vertex `i` to vertex `j`.

Steps:
1. **Initialization**:
   - Create a distance dictionary `dist` where each vertex initially has a distance of infinity to all other vertices.
   - Set the distance from a vertex to itself as zero.
   - Initialize distances between directly connected vertices based on edge weights in the graph.

2. **Iterative Update**:
   - For each vertex `k`, consider it as an intermediate point, and update the distance between each pair of vertices `(i, j)` by comparing the existing distance `dist[i][j]` with the distance obtained by going through `k` (i.e., `dist[i][k] + dist[k][j]`).
   - This process finds the shortest paths using intermediate vertices and gradually builds up the final shortest paths between all pairs.

3. **Result**:
   - After processing all vertices as intermediates, `dist[i][j]` contains the shortest distance from vertex `i` to vertex `j` for every pair `(i, j)`.

Key Points:
- Handles graphs with negative edge weights, as long as there are no negative weight cycles.
- If `dist[i][j]` is infinity after the algorithm, there is no path between `i` and `j`.

Example:
For the example graph, the output shows the shortest paths between each pair of vertices, or "No path" if they are not reachable.
"""
