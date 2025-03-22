from collections import defaultdict, deque

def find_connected_components(graph):
    visited = set()
    connected_components = []

    def bfs(start_node):
        queue = deque([start_node])
        component = []
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                component.append(node)
                queue.extend(graph[node] - visited)
        return component

    for node in graph:
        if node not in visited:
            connected_components.append(bfs(node))

    return connected_components

# Example graph (undirected)
graph = {
    1: {2, 3},
    2: {1, 4},
    3: {1},
    4: {2},
    5: {6},
    6: {5}
}

print(find_connected_components(graph))