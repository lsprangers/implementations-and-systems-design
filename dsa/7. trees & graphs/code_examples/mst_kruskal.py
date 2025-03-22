from unionfind import UnionFind

def kruskal_mst(vertices, edges):
    """
    Kruskal's MST Algorithm
    :param vertices: Number of vertices in the graph
    :param edges: List of edges (u, v, weight)
    :return: List of edges in the MST and the total weight
    """
    # Step 1: Sort edges by weight
    edges.sort(key=lambda x: x[2])

    # Step 2: Initialize Union-Find and MST
    uf = UnionFind(vertices)
    mst = []
    total_weight = 0

    # Step 3: Process edges
    for u, v, weight in edges:
        if uf.union(u, v):  # If u and v are not in the same component
            mst.append((u, v, weight))
            total_weight += weight

            # Stop if we have V - 1 edges in the MST
            if len(mst) == vertices - 1:
                break

    return mst, total_weight


# Example Usage
if __name__ == "__main__":
    # Example graph: 4 vertices (0, 1, 2, 3) and 5 edges
    vertices = 4
    edges = [
        (0, 1, 10),  # Edge from 0 to 1 with weight 10
        (0, 2, 6),   # Edge from 0 to 2 with weight 6
        (0, 3, 5),   # Edge from 0 to 3 with weight 5
        (1, 3, 15),  # Edge from 1 to 3 with weight 15
        (2, 3, 4)    # Edge from 2 to 3 with weight 4
    ]

    mst, total_weight = kruskal_mst(vertices, edges)
    print("Edges in MST:", mst)
    print("Total weight of MST:", total_weight)