import heapq
import random

class Vertex:
    def __init__(self, label):
        self.label = label

    def __lt__(self, other):
        return False

class Graph:
    def __init__(self):
        self.adjacency_list = {}
        self.edge_weights_distance = {}
        self.edge_weights_time = {}

    def add_vertex(self, vertex):
        self.adjacency_list[vertex.label] = []

    def add_edge(self, from_vertex, to_vertex, distance, initial_time):
        self.adjacency_list[from_vertex.label].append(to_vertex.label)
        self.edge_weights_distance[(from_vertex.label, to_vertex.label)] = distance
        self.edge_weights_time[(from_vertex.label, to_vertex.label)] = initial_time

    def update_edge_time(self, from_label, to_label, new_time):
        if (from_label, to_label) in self.edge_weights_time:
            self.edge_weights_time[(from_label, to_label)] = new_time

def dijkstra_shortest_path(graph, start_label, end_label, mode='distance'):
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

def get_real_time_traffic_data(from_label, to_label):
    return int(random.uniform(60, 300))

def update_graph_with_real_time_data(graph, vertices):
    for i, from_vertex in enumerate(vertices):
        for j, to_vertex in enumerate(vertices):
            if i != j:
                real_time_weight = get_real_time_traffic_data(from_vertex.label, to_vertex.label)
                graph.update_edge_time(from_vertex.label, to_vertex.label, real_time_weight)

def format_time(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes} minutes and {remaining_seconds} seconds"

print("Select the routing mode:")
print("1 - Shortest Distance")
print("2 - Shortest Time (considering traffic)")
user_choice = input("Enter 1 for Distance or 2 for Time: ")

if user_choice == '1':
    mode = 'distance'
elif user_choice == '2':
    mode = 'time'
else:
    print("Invalid input. Defaulting to distance mode.")
    mode = 'distance'

print(f"Calculating shortest path by {mode}...\n")

vA = Vertex("Restaurant")
vB = Vertex("Intersection1")
vC = Vertex("Intersection2")
vD = Vertex("Customer")
vE = Vertex("Intersection3")
vF = Vertex("Intersection4")
vG = Vertex("Intersection5")
vH = Vertex("Intersection6")

graph = Graph()
for v in [vA, vB, vC, vD, vE, vF, vG, vH]:
    graph.add_vertex(v)

graph.add_edge(vA, vB, distance=2, initial_time=180)
graph.add_edge(vB, vC, distance=1, initial_time=120)
graph.add_edge(vC, vD, distance=3, initial_time=240)
graph.add_edge(vA, vE, distance=3, initial_time=240)
graph.add_edge(vE, vF, distance=1, initial_time=90)
graph.add_edge(vF, vG, distance=1, initial_time=90)
graph.add_edge(vG, vD, distance=2, initial_time=150)
graph.add_edge(vB, vH, distance=1, initial_time=90)
graph.add_edge(vH, vF, distance=1, initial_time=120)
graph.add_edge(vF, vD, distance=1, initial_time=90)

if mode == 'time':
    update_graph_with_real_time_data(graph, [vA, vB, vC, vD, vE, vF, vG, vH])

shortest_path, total_cost = dijkstra_shortest_path(graph, "Restaurant", "Customer", mode=mode)
print(f"Shortest path from Restaurant to Customer by {mode}:", shortest_path)
if mode == 'time':
    print("Total travel time:", format_time(total_cost))
else:
    print("Total distance cost:", total_cost, "units")
