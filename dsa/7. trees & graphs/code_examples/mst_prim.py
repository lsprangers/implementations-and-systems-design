import heapq

def prim_mst(vertices, graph):
    """
    Prim's MST Algorithm
    :param vertices: Number of vertices in the graph
    :param graph: Adjacency list representation of the graph
    :return: List of edges in the MST and the total weight
    """
    # Initialize
    mst = []
    total_weight = 0
    visited = [False] * vertices
    min_heap = []

    # Start from arbitrary vertex
    visited[0] = True
    for neighbor, weight in graph[0]:
        heapq.heappush(min_heap, (weight, 0, neighbor))  # (weight, from, to)

    # Process the priority queue
    # This will start with smallest weighted edge that we found from 0
    while min_heap:
        weight, u, v = heapq.heappop(min_heap)

        if not visited[v]:
            # Add edge to MST
            visited[v] = True
            mst.append((u, v, weight))
            total_weight += weight

            # Add all edges from the new vertex to the heap
            for neighbor, edge_weight in graph[v]:
                if not visited[neighbor]:
                    heapq.heappush(min_heap, (edge_weight, v, neighbor))

    return mst, total_weight


# Example Usage
if __name__ == "__main__":
    # Example graph: 4 vertices (0, 1, 2, 3)
    # Adjacency list representation
    graph = {
        0: [(1, 10), (2, 6), (3, 5)],
        1: [(0, 10), (3, 15)],
        2: [(0, 6), (3, 4)],
        3: [(0, 5), (1, 15), (2, 4)]
    }

    vertices = 4
    mst, total_weight = prim_mst(vertices, graph)
    print("Edges in MST:", mst)
    print("Total weight of MST:", total_weight)