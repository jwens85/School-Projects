import heapq
import random

class Vertex:
    def __init__(self, VertexID):
        self.VertexID = VertexID

    def __lt__(self, other):
        return False

class Graph:
    def __init__(self):
        self.adjacency_list = {}
        self.edge_weights_distance = {}
        self.edge_weights_time = {}

    def add_vertex(self, vertex):
        self.adjacency_list[vertex.VertexID] = []

    def add_edge(self, from_vertex, to_vertex, distance, initial_time):
        self.adjacency_list[from_vertex.VertexID].append(to_vertex.VertexID)
        self.edge_weights_distance[(from_vertex.VertexID, to_vertex.VertexID)] = distance
        self.edge_weights_time[(from_vertex.VertexID, to_vertex.VertexID)] = initial_time

    def update_edge_time(self, from_VertexID, to_VertexID, new_time):
        if (from_VertexID, to_VertexID) in self.edge_weights_time:
            self.edge_weights_time[(from_VertexID, to_VertexID)] = new_time

def dijkstra_shortest_path(graph, start_VertexID, end_VertexID, mode='distance'):
    edge_weights = graph.edge_weights_distance if mode == 'distance' else graph.edge_weights_time

    distances = {VertexID: float('inf') for VertexID in graph.adjacency_list}
    predecessors = {VertexID: None for VertexID in graph.adjacency_list}
    distances[start_VertexID] = 0
    priority_queue = [(0, start_VertexID)]
    heapq.heapify(priority_queue)

    while priority_queue:
        current_distance, current_VertexID = heapq.heappop(priority_queue)
        if current_VertexID == end_VertexID:
            break

        for neighbor_VertexID in graph.adjacency_list[current_VertexID]:
            edge_weight = edge_weights.get((current_VertexID, neighbor_VertexID), float('inf'))
            alternative_path_distance = current_distance + edge_weight

            if alternative_path_distance < distances[neighbor_VertexID]:
                distances[neighbor_VertexID] = alternative_path_distance
                predecessors[neighbor_VertexID] = current_VertexID
                heapq.heappush(priority_queue, (alternative_path_distance, neighbor_VertexID))

    path = []
    current_VertexID = end_VertexID
    while current_VertexID is not None:
        path.insert(0, current_VertexID)
        current_VertexID = predecessors[current_VertexID]

    if path[0] == start_VertexID:
        formatted_path = "\n".join(path)
        return formatted_path, distances[end_VertexID]
    else:
        return "No path found", None

def get_real_time_traffic_data(from_VertexID, to_VertexID):
    return int(random.uniform(60, 300))

def update_graph_with_real_time_data(graph, vertices):
    for i, from_vertex in enumerate(vertices):
        for j, to_vertex in enumerate(vertices):
            if i != j:
                real_time_weight = get_real_time_traffic_data(from_vertex.VertexID, to_vertex.VertexID)
                graph.update_edge_time(from_vertex.VertexID, to_vertex.VertexID, real_time_weight)

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
print(f"Shortest path from Restaurant to Customer by {mode}:")
print(shortest_path)
if mode == 'time':
    print("Total travel time:", format_time(total_cost))
else:
    print("Total distance cost:", total_cost, "miles")
