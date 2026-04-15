# A single step in the search process. Every time an algorithm considers moving somewhere,
# it creates a Node for that position. Each Node remembers where it came from
# so we can trace the full path back to the start once the goal is found.
class Node:

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state     = state
        self.parent    = parent
        self.action    = action
        self.path_cost = path_cost
        self.depth     = (parent.depth + 1) if parent else 0

    def solution(self):
        actions = []
        node = self
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        return list(reversed(actions))

    def path(self):
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