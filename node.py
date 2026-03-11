
class Node:
    """
    Represents a single node in the search tree.

    Attributes:
        state     : (row, col) position in the maze
        parent    : parent Node (None for root)
        action    : action taken to reach this node ('UP', 'DOWN', etc.)
        path_cost : cumulative cost g(n) from start to this node
        depth     : depth in the search tree
    """

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state     = state
        self.parent    = parent
        self.action    = action
        self.path_cost = path_cost
        self.depth     = (parent.depth + 1) if parent else 0

    def solution(self):
        """Returns list of actions from root → this node."""
        actions = []
        node = self
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        return list(reversed(actions))

    def path(self):
        """Returns list of states from root → this node."""
        states = []
        node = self
        while node is not None:
            states.append(node.state)
            node = node.parent
        return list(reversed(states))

    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

    def __repr__(self):
        return (f"Node(state={self.state}, cost={self.path_cost}, "
                f"depth={self.depth}, action={self.action})")