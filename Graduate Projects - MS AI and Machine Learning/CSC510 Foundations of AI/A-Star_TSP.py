from simpleai.search.traditional import astar
from simpleai.search.models import SearchProblem

Distance_Matrix = {
    'Sandusky': {'Toledo': 67, 'Fremont': 27, 'Findlay': 66, 'Bowling Green': 85, 'Columbus': 106, 'Cleveland': 60, 'Dayton': 160, 'Akron': 90, 'Cincinnati': 200},
    'Toledo': {'Sandusky': 67, 'Fremont': 33, 'Findlay': 47, 'Bowling Green': 24, 'Columbus': 120, 'Cleveland': 115, 'Dayton': 150, 'Akron': 120, 'Cincinnati': 200},
    'Fremont': {'Sandusky': 27, 'Toledo': 33, 'Findlay': 49, 'Bowling Green': 32, 'Columbus': 100, 'Cleveland': 80, 'Dayton': 140, 'Akron': 70, 'Cincinnati': 180},
    'Findlay': {'Sandusky': 66, 'Toledo': 47, 'Fremont': 49, 'Bowling Green': 22, 'Columbus': 95, 'Cleveland': 130, 'Dayton': 90, 'Akron': 110, 'Cincinnati': 150},
    'Bowling Green': {'Sandusky': 85, 'Toledo': 24, 'Fremont': 32, 'Findlay': 22, 'Columbus': 110, 'Cleveland': 100, 'Dayton': 120, 'Akron': 100, 'Cincinnati': 170},
    'Columbus': {'Sandusky': 106, 'Toledo': 120, 'Fremont': 100, 'Findlay': 95, 'Bowling Green': 110, 'Cleveland': 140, 'Dayton': 70, 'Akron': 125, 'Cincinnati': 110},
    'Cleveland': {'Sandusky': 60, 'Toledo': 115, 'Fremont': 80, 'Findlay': 130, 'Bowling Green': 100, 'Columbus': 140, 'Dayton': 200, 'Akron': 40, 'Cincinnati': 250},
    'Dayton': {'Sandusky': 160, 'Toledo': 150, 'Fremont': 140, 'Findlay': 90, 'Bowling Green': 120, 'Columbus': 70, 'Cleveland': 200, 'Akron': 150, 'Cincinnati': 50},
    'Akron': {'Sandusky': 90, 'Toledo': 120, 'Fremont': 70, 'Findlay': 110, 'Bowling Green': 100, 'Columbus': 125, 'Cleveland': 40, 'Dayton': 150, 'Cincinnati': 220},
    'Cincinnati': {'Sandusky': 200, 'Toledo': 200, 'Fremont': 180, 'Findlay': 150, 'Bowling Green': 170, 'Columbus': 110, 'Cleveland': 250, 'Dayton': 50, 'Akron': 220}
}

Ohio_Cities = list(Distance_Matrix.keys())

class OhioTSP(SearchProblem):
    def __init__(self, start_city='Sandusky'):
        super().__init__()
        self.initial_state = (start_city, tuple([start_city]))
        self.start_city = start_city

    def actions(self, state):
        current_city, visited_cities = state
        if len(visited_cities) == len(Ohio_Cities) and current_city != self.start_city:
            return [self.start_city]
        return [city for city in Ohio_Cities if city not in visited_cities]

    def result(self, state, next_city):
        current_city, visited_cities = state
        return next_city, tuple(visited_cities + (next_city,))

    def is_goal(self, state):
        current_city, visited_cities = state
        return len(visited_cities) == len(Ohio_Cities) + 1 and current_city == self.start_city

    def cost(self, state, next_city, new_state):
        current_city, _ = state
        return Distance_Matrix[current_city][next_city]

    def heuristic(self, state):
        current_city, visited_cities = state
        remaining_cities = [city for city in Ohio_Cities if city not in visited_cities]
        if current_city not in Distance_Matrix:
            return float('inf')
        if not remaining_cities:
            return Distance_Matrix[current_city].get(self.start_city, float('inf'))
        return min(Distance_Matrix[current_city].get(city, float('inf')) for city in remaining_cities)

def main():
    start_city = 'Sandusky'
    print(f"Starting journey from {start_city}")

    tsp_problem = OhioTSP(start_city)
    best_route = astar(tsp_problem)

    if best_route is None:
        print("No route found!")
    else:
        path = best_route.path()
        total_miles = sum(tsp_problem.cost(path[i][1], path[i+1][0], path[i+1][1]) for i in range(len(path) - 1))
        print(f"Optimal route: {' -> '.join(state[0] for _, state in path)}")
        print(f"Total travel distance: {total_miles} miles")

if __name__ == '__main__':
    main()
