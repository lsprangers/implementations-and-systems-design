from collections import defaultdict, deque

def kahn_topological_sort(vertices, edges):
    """
    Kahn's Algorithm for Topological Sorting
    :param vertices: Number of vertices in the graph
    :param edges: List of directed edges (u, v) where u -> v
    :return: List representing the topological order of the graph
    """
    # Step 1: Initialize in-degree and adjacency list
    in_degree = {i: 0 for i in range(vertices)}
    adjacency_list = defaultdict(list)

    for u, v in edges:
        adjacency_list[u].append(v)
        in_degree[v] += 1

    # Step 2: Collect nodes with in-degree 0
    zero_in_degree = deque([node for node in in_degree if in_degree[node] == 0])

    # Step 3: Perform topological sort
    topo_order = []
    while zero_in_degree:
        current = zero_in_degree.popleft()
        topo_order.append(current)

        # Reduce in-degree of neighbors
        for neighbor in adjacency_list[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree.append(neighbor)

    # If topo_order doesn't contain all vertices, there is a cycle
    if len(topo_order) != vertices:
        raise ValueError("The graph contains a cycle and cannot be topologically sorted.")

    return topo_order