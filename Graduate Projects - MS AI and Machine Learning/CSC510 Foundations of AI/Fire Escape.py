import threading
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simpleai.search import SearchProblem, astar

# Grid Constants
EMPTY = 0
WALL = 1
EXIT = 2
FIRE = 3
PLAYER = 4

# Grid Dimensions
GRID_WIDTH = 8
GRID_HEIGHT = 8

# Probability of initial fire placement
FIRE_SPAWN_PROB = 0.1

# Directions (Up, Down, Left, Right)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generate a Random Building Layout
def generate_building():
    grid = [[EMPTY for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

    # Place walls randomly
    for _ in range(int(GRID_WIDTH * GRID_HEIGHT * 0.2)):  # 20% walls
        x, y = random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1)
        grid[y][x] = WALL

    # Place exits at the edges
    exits = []
    for _ in range(2):
        x, y = random.choice([(0, random.randint(0, GRID_HEIGHT-1)),
                              (GRID_WIDTH-1, random.randint(0, GRID_HEIGHT-1)),
                              (random.randint(0, GRID_WIDTH-1), 0),
                              (random.randint(0, GRID_WIDTH-1), GRID_HEIGHT-1)])
        grid[y][x] = EXIT
        exits.append((x, y))

    # Place fire randomly
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if grid[y][x] == EMPTY and random.random() < FIRE_SPAWN_PROB:
                grid[y][x] = FIRE

    # Set player start position
    while True:
        px, py = random.randint(1, GRID_WIDTH-2), random.randint(1, GRID_HEIGHT-2)
        if grid[py][px] == EMPTY:
            grid[py][px] = PLAYER
            player_pos = (px, py)
            break

    return grid, player_pos, exits

# A* Search Problem
class EscapeBuildingProblem(SearchProblem):
    def __init__(self, grid, player_pos, exits):
        self.grid = grid
        self.initial_state = player_pos
        self.exits = exits
        super().__init__(initial_state=self.initial_state)

    def actions(self, state):
        x, y = state
        valid_moves = []
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and self.grid[ny][nx] in (EMPTY, EXIT):
                valid_moves.append((dx, dy))
        return valid_moves

    def result(self, state, action):
        x, y = state
        dx, dy = action
        return (x + dx, y + dy)

    def is_goal(self, state):
        return state in self.exits

    def cost(self, state1, action, state2):
        return 1  # Constant movement cost

    def heuristic(self, state):
        x, y = state
        return min(abs(x - ex) + abs(y - ey) for ex, ey in self.exits)

# Convert grid to numpy array for visualization
def convert_grid_to_array(grid):
    return np.array(grid)

# Matplotlib Animation Function
def animate_escape(grid, path):
    fig, ax = plt.subplots()

    # Define colors for each type of cell
    cmap = plt.colormaps.get_cmap("coolwarm")
    grid_array = convert_grid_to_array(grid)

    # Show static grid (Fire, Walls, Exit)
    im = ax.imshow(grid_array, cmap=cmap, vmin=0, vmax=4)

    # Track player movement with a yellow dot
    player_dot, = ax.plot([], [], 'yo', markersize=10)

    def update(frame):
        x, y = path[frame][1]  # Extract position from path
        player_dot.set_data(x, y)  # Update player's position

    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=500, repeat=False)

    # Add colorbar for legend
    plt.colorbar(im, ticks=[0, 1, 2, 3, 4], label="Legend: 0=Empty, 1=Wall, 2=Exit, 3=Fire, 4=Player")
    plt.title("Burning Building Escape Simulation")
    plt.show()

# Timeout Wrapper (Windows Compatible)
class Timeout:
    def __init__(self, seconds=5, error_message="A* search timeout! No solution found."):
        self.seconds = seconds
        self.error_message = error_message
        self.timer = None

    def __enter__(self):
        self.timer = threading.Timer(self.seconds, self.raise_timeout)
        self.timer.start()

    def raise_timeout(self):
        raise TimeoutError(self.error_message)

    def __exit__(self, exc_type, exc_value, traceback):
        self.timer.cancel()  # Stop the timer if A* completes in time

# Main Simulation
def main():
    grid, player_pos, exits = generate_building()
    problem = EscapeBuildingProblem(grid, player_pos, exits)

    print("ESCAPE THE BURNING BUILDING!")
    print("Finding the fastest route...\n")

    # Use Windows-compatible timeout
    try:
        with Timeout(5):  # Timeout after 5 seconds
            solution = astar(problem)
        print("DEBUG: A* search completed")  # Check if A* finishes
    except TimeoutError:
        print("A* search timeout! No solution found.")
        return

    # Ensure solution exists before proceeding
    if solution:
        path = solution.path()
        print(f"Found an escape route in {len(path)-1} steps!\n")
    else:
        print("No escape possible! Trapped!\n")
        return

    # Visualize escape route
    animate_escape(grid, path)

    if grid[player_pos[1]][player_pos[0]] == EXIT:
        print("You escaped safely!\n")

if __name__ == "__main__":
    main()
