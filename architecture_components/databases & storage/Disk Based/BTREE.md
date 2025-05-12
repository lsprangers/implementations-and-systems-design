# BTree
B-Tree's are a durable, on disk, database storage structure 

- B-Trees are self-balancing tree data structures that maintain sorted data and allow efficient operations like:
    - Point Lookups: $O(\log n)$ time complexity.
    - Range Queries: $O(\log n + k)$, where $k$ is the number of matching results
- B-Trees are commonly used in relational databases for indexing columns, especially when range queries are required

## Implementation
