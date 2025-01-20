## Leet
This is just a dump of leetcode problems and comments

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
        consistent RAFT cluster to back a distributed, partitioned, KV Store