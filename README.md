## Design Systems
Building specific systems based utilizing [Typical Resources](#typical-resources)

Some examples include URL shortener, Youtube, Youtube Search, Top K Heavy Hitters, etc...

### Typical Resources
These are discussions and implementations of "typical" resources like distributed KV store, front end load balancer + metadata, distributed queue, etc...

## Implementations
This is a folder of generic implementations of things like
- "A Key Value store with transactions"
- "A balanced binary search tree from scratch"
- "ML feature store architecture diagram"

### KV Store with Transactions
- Just set one up from a challenge I saw on LeetCode
    - Getting the operations to all be O(1) is relatively easy, but the next 
        parts of the implementation would be multi-threading
    - Goes hand in hand with the RAFT repository where we show how to create a strongly
        consistent RAFT cluster to back a distributed, sharded, KV Store

### Deque
-  Creating a deque from linked lists, and doing things like sorting, traversal, etc...

### Merge Sort
- Following from needing merge sort in deque sort just implemented it here

## Leet
This is just a dump of leetcode problems and comments