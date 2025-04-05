class Vertex:
  def __init__(self, label):
      self.label = label
      self.distance = float('inf')
      self.pred_vertex = None

class Graph:
  def __init__(self):
      self.adjacency_list = {}
      self.edge_weights = {}

  def add_vertex(self, vertex):
      self.adjacency_list[vertex] = []

  def add_edge(self, from_vertex, to_vertex, weight):
      self.adjacency_list[from_vertex].append(to_vertex)
      self.edge_weights[(from_vertex, to_vertex)] = weight

def dijkstra_shortest_path(g, start_vertex, end_vertex):
  unvisited_queue = []
  for vertex in g.adjacency_list:
      vertex.distance = float('inf')
      vertex.pred_vertex = None
      unvisited_queue.append(vertex)

  start_vertex.distance = 0

  while len(unvisited_queue) > 0:
      smallest_index = 0
      for i in range(1, len(unvisited_queue)):
          if unvisited_queue[i].distance < unvisited_queue[smallest_index].distance:
              smallest_index = i
      current_vertex = unvisited_queue.pop(smallest_index)

      for adj_vertex in g.adjacency_list[current_vertex]:
          edge_weight = g.edge_weights[(current_vertex, adj_vertex)]
          alternative_path_distance = current_vertex.distance + edge_weight

          if alternative_path_distance < adj_vertex.distance:
              adj_vertex.distance = alternative_path_distance
              adj_vertex.pred_vertex = current_vertex

  path = ''
  current_vertex = end_vertex
  while current_vertex is not start_vertex:
      if current_vertex is None:
          return "No path found"
      path = ' -> ' + str(current_vertex.label) + path
      current_vertex = current_vertex.pred_vertex
  path = start_vertex.label + path  

  return path

vA = Vertex("A")
vB = Vertex("B")
vC = Vertex("C")
vD = Vertex("D")
vE = Vertex("E")

graph = Graph()
for v in [vA, vB, vC, vD, vE]:
  graph.add_vertex(v)

graph.add_edge(vA, vB, 4)
graph.add_edge(vA, vC, 1)
graph.add_edge(vC, vB, 2)
graph.add_edge(vB, vD, 5)
graph.add_edge(vC, vD, 8)
graph.add_edge(vD, vE, 3)
graph.add_edge(vC, vE, 10)

shortest_path = dijkstra_shortest_path(graph, vA, vE)
print("Shortest path from A to E:", shortest_path)
