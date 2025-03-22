import heapq

def dijkstra(graph, source):
    """
    Dijkstra's Algorithm to find the shortest path from a source to all other vertices.
    :param graph: Adjacency list representation of the graph {node: [(neighbor, weight), ...]}
    :param source: The starting vertex
    :return: A tuple (distances, previous_nodes)
             distances: Dictionary of shortest distances from the source to each vertex
             previous_nodes: Dictionary to reconstruct the shortest path
    """
    # Initialize distances and priority queue
    distances = {vertex: float('inf') for vertex in graph}
    distances[source] = 0
    previous_nodes = {vertex: None for vertex in graph}
    min_heap = [(0, source)]  # (distance, vertex)

    while min_heap:
        # O(log V)
        current_distance, current_vertex = heapq.heappop(min_heap)

        # Skip if the current distance is not optimal
        if current_distance > distances[current_vertex]:
            continue

        # Explore neighbors
        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight

            # If a shorter path is found
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_vertex
                # O(log V)
                heapq.heappush(min_heap, (distance, neighbor))

    return distances, previous_nodes


def reconstruct_path(previous_nodes, target):
    """
    Reconstruct the shortest path from the source to the target.
    :param previous_nodes: Dictionary of previous nodes from Dijkstra's algorithm
    :param target: The target vertex
    :return: List of vertices representing the shortest path
    """
    path = []
    while target is not None:
        path.append(target)
        target = previous_nodes[target]
    return path[::-1]  # Reverse the path


if __name__ == "__main__":
    # Example graph: Adjacency list representation
    # Here we'd want 0 - 2 - 1 - 3 for total weight of 1 + 2 + 1 = 4
    graph = {
        0: [(1, 4), (2, 1)],
        1: [(3, 1)],
        2: [(1, 2), (3, 5)],
        3: []
    }

    source = 0
    target = 3

    # Run Dijkstra's algorithm
    distances, previous_nodes = dijkstra(graph, source)

    # Output results
    print("Shortest distances from source:", distances)
    print("Shortest path to target:", reconstruct_path(previous_nodes, target))