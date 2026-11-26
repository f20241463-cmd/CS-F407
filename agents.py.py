"""
Goal-Based Agent for the Warehouse Navigation Problem
=====================================================

AGENT DESIGN (PEAS / goal-based architecture)
---------------------------------------------
Environment : 2-D grid warehouse (static, fully observable, deterministic,
              discrete). '#' = obstacle, '.' = free, 'S' = start, 'G' = goal.
State       : the vehicle's current (row, col) position.
Goal        : reach the cell marked 'G'.
Actions     : Up, Down, Left, Right (one grid square each).
Decision    : Breadth-First Search (BFS) over the state space. The agent
              builds a *plan* (a sequence of actions) that achieves the goal,
              then executes it.

WHY BREADTH-FIRST SEARCH?
-------------------------
* Every move has the same cost (one square), so BFS is OPTIMAL: it returns
  a shortest path.
* BFS is COMPLETE: if any path exists it will find it, and if none exists
  it terminates and reports that fact (the visited set prevents loops).
* The grid is small, so BFS's memory use O(rows*cols) is not a problem.
  (For much larger maps, A* with a Manhattan-distance heuristic would
  explore far fewer cells.)
"""

from collections import deque

WAREHOUSE_MAP = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################",
]

# Action name -> (row change, column change)
ACTIONS = {
    "Up":    (-1, 0),
    "Down":  (1, 0),
    "Left":  (0, -1),
    "Right": (0, 1),
}


class WarehouseEnvironment:
    """The environment: stores the grid and answers questions about it."""

    def __init__(self, layout):
        self.grid = [list(row) for row in layout]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.start = self._find("S")
        self.goal = self._find("G")

    def _find(self, symbol):
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell == symbol:
                    return (r, c)
        raise ValueError(f"Symbol '{symbol}' not found in map")

    def is_free(self, pos):
        r, c = pos
        return (0 <= r < self.rows and 0 <= c < self.cols
                and self.grid[r][c] != "#")

    def successors(self, pos):
        """Transition model: yield (action, next_position) for legal moves."""
        for name, (dr, dc) in ACTIONS.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if self.is_free(nxt):
                yield name, nxt

    def display(self, path=None):
        """Return the map as text, marking the path with '*'."""
        out = [row[:] for row in self.grid]
        for (r, c) in (path or []):
            if out[r][c] == ".":
                out[r][c] = "*"
        return "\n".join("".join(row) for row in out)


class GoalBasedAgent:
    """Goal-based agent: searches for a sequence of actions reaching the goal."""

    def __init__(self, environment):
        self.env = environment
        self.state = environment.start      # current state (percept)
        self.goal = environment.goal        # explicit goal
        self.nodes_expanded = 0

    def goal_test(self, state):
        return state == self.goal

    def plan(self):
        """BFS. Returns (list of positions, list of actions) or None."""
        frontier = deque([self.state])
        # parent[state] = (previous_state, action_taken); doubles as 'visited'
        parent = {self.state: None}

        while frontier:
            current = frontier.popleft()
            self.nodes_expanded += 1

            if self.goal_test(current):
                return self._reconstruct(parent, current)

            for action, nxt in self.env.successors(current):
                if nxt not in parent:
                    parent[nxt] = (current, action)
                    frontier.append(nxt)
        return None     # frontier exhausted: goal unreachable

    @staticmethod
    def _reconstruct(parent, state):
        positions, actions = [], []
        while parent[state] is not None:
            positions.append(state)
            state, action = parent[state]
            actions.append(action)
        positions.append(state)
        positions.reverse()
        actions.reverse()
        return positions, actions

    def execute(self, actions):
        """Carry out the plan, updating the agent's state step by step."""
        for action in actions:
            dr, dc = ACTIONS[action]
            self.state = (self.state[0] + dr, self.state[1] + dc)
        return self.goal_test(self.state)


def solve(layout):
    env = WarehouseEnvironment(layout)
    agent = GoalBasedAgent(env)
    print("Warehouse map:")
    print(env.display())
    print(f"\nStart: {env.start}   Goal: {env.goal}")

    result = agent.plan()
    if result is None:
        print("\nNo path exists from S to G.")
        return None

    positions, actions = result
    print(f"\nPath found: {len(actions)} moves "
          f"({agent.nodes_expanded} cells expanded)")
    print("Actions :", " ".join(actions))
    print("Positions:", positions)
    print("\nMap with path marked by '*':")
    print(env.display(positions))
    print("\nGoal reached by executing plan:", agent.execute(actions))
    return actions


def run_tests():
    """Simple validation tests."""
    # 1. Given map: path must be valid and avoid obstacles
    env = WarehouseEnvironment(WAREHOUSE_MAP)
    agent = GoalBasedAgent(env)
    positions, actions = agent.plan()
    assert positions[0] == env.start and positions[-1] == env.goal
    assert all(env.is_free(p) for p in positions)
    assert all(abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1
               for a, b in zip(positions, positions[1:]))
    assert agent.execute(actions)

    # 2. Unreachable goal: must report no path
    blocked = ["#######",
               "#S.#.G#",
               "#######"]
    assert GoalBasedAgent(WarehouseEnvironment(blocked)).plan() is None

    # 3. Start next to goal
    tiny = ["####", "#SG#", "####"]
    assert GoalBasedAgent(WarehouseEnvironment(tiny)).plan()[1] == ["Right"]
    print("All tests passed.")


if __name__ == "__main__":
    solve(WAREHOUSE_MAP)
    print("\n--- Tests ---")
    run_tests()
    print("\n--- Unreachable goal demo ---")
    solve(["#######", "#S.#.G#", "#######"])
