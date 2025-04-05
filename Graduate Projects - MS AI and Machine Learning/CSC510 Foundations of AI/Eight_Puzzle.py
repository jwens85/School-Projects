#!/usr/bin/env python3
"""
8-Puzzle Solver using A* Search with SimpleAI
Usage: python eight_puzzle.py
"""

from simpleai.search import SearchProblem, astar


def index_to_coord(index):
    """Convert a linear index to (row, col) for a 3x3 grid."""
    return divmod(index, 3)


def manhattan_distance(state, goal_state):
    """Calculate the Manhattan distance for the 8-puzzle state."""
    distance = 0
    for tile in range(1, 9):  # ignore the blank (assumed to be 0)
        current_index = state.index(tile)
        goal_index = goal_state.index(tile)
        current_row, current_col = index_to_coord(current_index)
        goal_row, goal_col = index_to_coord(goal_index)
        distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance


class EightPuzzleProblem(SearchProblem):
    def __init__(self, initial, goal):
        # SimpleAI requires 'initial_state' to be set
        self.initial_state = initial
        self.goal = goal

    def actions(self, state):
        """Return possible moves as a list of actions."""
        # Find the index of the blank (0)
        blank_index = state.index(0)
        row, col = index_to_coord(blank_index)
        possible_actions = []

        if row > 0:
            possible_actions.append("UP")
        if row < 2:
            possible_actions.append("DOWN")
        if col > 0:
            possible_actions.append("LEFT")
        if col < 2:
            possible_actions.append("RIGHT")

        return possible_actions

    def result(self, state, action):
        """Return the new state after performing the given action."""
        new_state = list(state)
        blank_index = state.index(0)

        if action == "UP":
            target_index = blank_index - 3
        elif action == "DOWN":
            target_index = blank_index + 3
        elif action == "LEFT":
            target_index = blank_index - 1
        elif action == "RIGHT":
            target_index = blank_index + 1
        else:
            raise ValueError("Unknown action: " + action)

        # Swap the blank with the target tile
        new_state[blank_index], new_state[target_index] = new_state[target_index], new_state[blank_index]
        return tuple(new_state)

    def is_goal(self, state):
        return state == self.goal

    def cost(self, state, action, state2):
        return 1  # Uniform cost for each move

    def heuristic(self, state):
        return manhattan_distance(state, self.goal)


def main():
    # Define the initial and goal states (0 represents the blank space)
    initial_state = (1, 2, 3,
                     4, 0, 6,
                     7, 5, 8)

    goal_state = (1, 2, 3,
                  4, 5, 6,
                  7, 8, 0)

    print("Initial State:")
    print(initial_state)
    print("Goal State:")
    print(goal_state)
    print("Solving...")

    # Create the problem instance and perform A* search
    problem = EightPuzzleProblem(initial_state, goal_state)
    result = astar(problem)

    if result is None:
        print("No solution found!")
    else:
        path = result.path()
        print("Solution found in", len(path) - 1, "steps.")  # Excluding the initial state

        # Skip the first step (initial state) and only show actual actions
        for action, state in path[1:]:
            print("Action:", action)
            print("State:", state)
            print("------")


if __name__ == '__main__':
    main()
