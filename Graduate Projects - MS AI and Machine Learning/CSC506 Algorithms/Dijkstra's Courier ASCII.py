import heapq
import random

class Vertex:
    def __init__(self, label):
        self.label = label  # Unique identifier for the vertex

    def __lt__(self, other):
        return False  # Placeholder for priority queue compatibility

class Graph:
    def __init__(self):
        self.adjacency_list = {}  # Dictionary with labels as keys and adjacency lists as values
        self.edge_weights_distance = {}  # Static weights representing distance
        self.edge_weights_time = {}  # Dynamic weights representing travel time in seconds

    def add_vertex(self, vertex):
        self.adjacency_list[vertex.label] = []  # Initialize adjacency list with vertex label as key

    def add_edge(self, from_vertex, to_vertex, distance, initial_time):
        # Add directed edge with both distance and time weights
        self.adjacency_list[from_vertex.label].append(to_vertex.label)
        self.edge_weights_distance[(from_vertex.label, to_vertex.label)] = distance
        self.edge_weights_time[(from_vertex.label, to_vertex.label)] = initial_time

    def update_edge_time(self, from_label, to_label, new_time):
        # Update only the time-based weight with new real-time data
        if (from_label, to_label) in self.edge_weights_time:
            self.edge_weights_time[(from_label, to_label)] = new_time

def dijkstra_shortest_path(graph, start_label, end_label, mode='distance'):
    # Use the appropriate edge weights based on mode
    edge_weights = graph.edge_weights_distance if mode == 'distance' else graph.edge_weights_time

    distances = {label: float('inf') for label in graph.adjacency_list}
    predecessors = {label: None for label in graph.adjacency_list}
    distances[start_label] = 0
    priority_queue = [(0, start_label)]
    heapq.heapify(priority_queue)

    while priority_queue:
        current_distance, current_label = heapq.heappop(priority_queue)
        if current_label == end_label:
            break

        for neighbor_label in graph.adjacency_list[current_label]:
            edge_weight = edge_weights.get((current_label, neighbor_label), float('inf'))
            alternative_path_distance = current_distance + edge_weight

            if alternative_path_distance < distances[neighbor_label]:
                distances[neighbor_label] = alternative_path_distance
                predecessors[neighbor_label] = current_label
                heapq.heappush(priority_queue, (alternative_path_distance, neighbor_label))

    path = []
    current_label = end_label
    while current_label is not None:
        path.insert(0, current_label)
        current_label = predecessors[current_label]

    if path[0] == start_label:
        return ' -> '.join(path), distances[end_label]
    else:
        return "No path found", None

# Real-time Traffic Data Integration (Simulated for Example)
def get_real_time_traffic_data(from_label, to_label):
    # Simulate an API call to fetch real-time traffic data, return time in seconds
    return int(random.uniform(60, 300))  # Random travel time between 1 to 5 minutes in seconds

def update_graph_with_real_time_data(graph, vertices):
    for i, from_vertex in enumerate(vertices):
        for j, to_vertex in enumerate(vertices):
            if i != j:
                real_time_weight = get_real_time_traffic_data(from_vertex.label, to_vertex.label)
                graph.update_edge_time(from_vertex.label, to_vertex.label, real_time_weight)

def format_time(seconds):
    # Convert total seconds to minutes and seconds
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes} minutes and {remaining_seconds} seconds"

# Prompt user for mode choice
print("Select the routing mode:")
print("1 - Shortest Distance")
print("2 - Shortest Time (considering traffic)")
user_choice = input("Enter 1 for Distance or 2 for Time: ")

# Set the mode based on user input
if user_choice == '1':
    mode = 'distance'
elif user_choice == '2':
    mode = 'time'
else:
    print("Invalid input. Defaulting to distance mode.")
    mode = 'distance'

print(f"Calculating shortest path by {mode}...\n")

# Create vertices
vA = Vertex("Restaurant")
vB = Vertex("Intersection1")
vC = Vertex("Intersection2")
vD = Vertex("Customer")
vE = Vertex("Intersection3")
vF = Vertex("Intersection4")
vG = Vertex("Intersection5")
vH = Vertex("Intersection6")

# Create a graph and add vertices
graph = Graph()
for v in [vA, vB, vC, vD, vE, vF, vG, vH]:
    graph.add_vertex(v)

# Add edges with both distance and initial time weights in seconds
graph.add_edge(vA, vB, distance=2, initial_time=180)  # 3 minutes
graph.add_edge(vB, vC, distance=1, initial_time=120)  # 2 minutes
graph.add_edge(vC, vD, distance=3, initial_time=240)  # 4 minutes
graph.add_edge(vA, vE, distance=3, initial_time=240)  # 4 minutes
graph.add_edge(vE, vF, distance=1, initial_time=90)   # 1.5 minutes
graph.add_edge(vF, vG, distance=1, initial_time=90)   # 1.5 minutes
graph.add_edge(vG, vD, distance=2, initial_time=150)  # 2.5 minutes
graph.add_edge(vB, vH, distance=1, initial_time=90)   # 1.5 minutes
graph.add_edge(vH, vF, distance=1, initial_time=120)  # 2 minutes
graph.add_edge(vF, vD, distance=1, initial_time=90)   # 1.5 minutes

# Update edge weights with real-time traffic data for time mode
if mode == 'time':
    update_graph_with_real_time_data(graph, [vA, vB, vC, vD, vE, vF, vG, vH])

# Calculate and print the shortest path from Restaurant to Customer
shortest_path, total_cost = dijkstra_shortest_path(graph, "Restaurant", "Customer", mode=mode)
print(f"Shortest path from Restaurant to Customer by {mode}:", shortest_path)
if mode == 'time':
    print("Total travel time:", format_time(total_cost))
else:
    print("Total distance cost:", total_cost, "units")
