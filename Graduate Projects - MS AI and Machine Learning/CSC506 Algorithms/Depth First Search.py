# Depth-first search function
def depth_first_search(graph, start_vertex, visit_function):
    vertex_stack = [start_vertex]  # Initialize stack with the starting vertex
    visited_set = set()  # Set to track visited vertices to avoid re-processing

    while len(vertex_stack) > 0:  # Continue while there are vertices to explore
        current_vertex = vertex_stack.pop()  # Pop the last vertex added to the stack (LIFO order)
        if current_vertex not in visited_set:  # Process only if vertex has not been visited
            visit_function(current_vertex)  # Apply the visit function to the current vertex
            visited_set.add(current_vertex)  # Mark the current vertex as visited

            # Add all adjacent vertices to the stack for further exploration
            for adjacent_vertex in graph.adjacency_list[current_vertex]:
                vertex_stack.append(adjacent_vertex)  # Push adjacent vertices onto the stack

"""
Explanation:
1. `depth_first_search` function:
   - Performs a depth-first traversal of the graph using a stack (LIFO order).
   - Uses a `visit_function` to process each vertex as it’s visited.

2. Parameters:
   - `graph`: The graph to search, represented by an adjacency list.
   - `start_vertex`: The starting vertex for the DFS traversal.
   - `visit_function`: A function applied to each vertex when it is visited (e.g., for printing or processing).

3. Functionality:
   - `vertex_stack`: A stack initialized with `start_vertex` for tracking the vertices to explore.
   - `visited_set`: Keeps track of visited vertices to avoid cycles and re-visiting.

4. Loop operation:
   - The loop continues as long as there are vertices in `vertex_stack`.
   - Each iteration pops a vertex from the stack. If it hasn’t been visited, it’s processed by `visit_function` and added to `visited_set`.
   - All adjacent vertices of `current_vertex` are pushed onto the stack, so they can be explored in subsequent iterations.

This approach provides a depth-first traversal of the graph, visiting all reachable vertices from `start_vertex`.
"""
