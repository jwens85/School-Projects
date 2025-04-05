import matplotlib.pyplot as plt
import heapq
import random

class Vertex:
    def __init__(self, label):
        self.label = label

class Graph:
    def __init__(self):
        self.edge_weights_time = {}
        self.adjacency_list = {}

    def add_edge(self, from_vertex, to_vertex, initial_time):
        if from_vertex not in self.adjacency_list:
            self.adjacency_list[from_vertex] = []
        if to_vertex not in self.adjacency_list:
            self.adjacency_list[to_vertex] = []
        self.adjacency_list[from_vertex].append(to_vertex)
        self.edge_weights_time[(from_vertex, to_vertex)] = initial_time

    def update_edge_time(self, from_vertex, to_vertex, new_time):
        if (from_vertex, to_vertex) in self.edge_weights_time:
            self.edge_weights_time[(from_vertex, to_vertex)] = new_time

def get_real_time_traffic_data():
    return int(random.uniform(60, 300))

def update_graph_with_real_time_data(graph, edges):
    for from_vertex, to_vertex in edges:
        real_time_weight = get_real_time_traffic_data()
        graph.update_edge_time(from_vertex, to_vertex, real_time_weight)

def format_short_time(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes}m{remaining_seconds}s"

def dijkstra_shortest_path(graph, start_vertex, end_vertex):
    distances = {vertex: float('inf') for vertex in graph.adjacency_list}
    predecessors = {vertex: None for vertex in graph.adjacency_list}
    distances[start_vertex] = 0

    priority_queue = [(0, start_vertex)]
    heapq.heapify(priority_queue)

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_vertex == end_vertex:
            break

        for neighbor in graph.adjacency_list[current_vertex]:
            edge_weight = graph.edge_weights_time.get((current_vertex, neighbor), float('inf'))
            alternative_path_distance = current_distance + edge_weight

            if alternative_path_distance < distances[neighbor]:
                distances[neighbor] = alternative_path_distance
                predecessors[neighbor] = current_vertex
                heapq.heappush(priority_queue, (alternative_path_distance, neighbor))

    # Reconstruct the shortest path
    path = []
    current_vertex = end_vertex
    while current_vertex is not None:
        path.insert(0, current_vertex)
        current_vertex = predecessors[current_vertex]

    if path[0] == start_vertex:
        return path, distances[end_vertex]
    else:
        return None, None  # No path found

def draw_graph(graph, path=None):
    # Define positions for each vertex in a grid layout
    pos = {
        "Restaurant": (0, 2),
        "Intersection1": (1, 2),
        "Intersection2": (2, 2),
        "Customer": (3, 2),
        "Intersection3": (1, 3),
        "Intersection4": (2, 3),
        "Intersection5": (2, 1),
        "Intersection6": (1, 1)
    }

    # Create the plot
    fig, ax = plt.subplots()

    # Draw nodes (intersections and key points)
    for vertex, (x, y) in pos.items():
        ax.plot(x, y, 'o', markersize=10, color="skyblue")
        ax.text(x, y + 0.1, vertex, ha='center', fontsize=9, color="black")

    # Draw edges with travel times as labels
    for (from_vertex, to_vertex), time in graph.edge_weights_time.items():
        x_from, y_from = pos[from_vertex]
        x_to, y_to = pos[to_vertex]
        color = "red" if path and (from_vertex, to_vertex) in zip(path, path[1:]) else "gray"
        ax.annotate("",
                    xy=(x_to, y_to), xycoords='data',
                    xytext=(x_from, y_from), textcoords='data',
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        # Label the edge with travel time
        ax.text((x_from + x_to) / 2, (y_from + y_to) / 2, format_short_time(time),
                ha='center', fontsize=8, color="red")

    # Set plot limits and title
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0.5, 3.5)
    ax.set_aspect('equal')
    plt.title("Simplified Delivery Route Network with Real-Time Traffic Times")
    plt.axis("off")
    plt.show()

# Create vertices and graph
graph = Graph()
vertices = ["Restaurant", "Intersection1", "Intersection2", "Customer", "Intersection3", "Intersection4", "Intersection5", "Intersection6"]

# Define edges with initial times
edges = [
    ("Restaurant", "Intersection1", 180),
    ("Intersection1", "Intersection2", 120),
    ("Intersection2", "Customer", 150),
    ("Intersection1", "Intersection3", 240),
    ("Intersection3", "Intersection4", 90),
    ("Intersection4", "Intersection2", 180),
    ("Intersection1", "Intersection6", 150),
    ("Intersection6", "Intersection5", 120),
    ("Intersection5", "Intersection2", 200)
]

# Add edges to the graph
for from_vertex, to_vertex, initial_time in edges:
    graph.add_edge(from_vertex, to_vertex, initial_time)

# Update edges with real-time traffic data
update_graph_with_real_time_data(graph, [(e[0], e[1]) for e in edges])

# Calculate the shortest path from Restaurant to Customer
path, total_time = dijkstra_shortest_path(graph, "Restaurant", "Customer")

if path:
    print("Shortest path:", " -> ".join(path))
    print("Total travel time:", format_short_time(total_time))
else:
    print("No path found from Restaurant to Customer.")

# Draw the graph with the shortest path highlighted
draw_graph(graph, path=path)
