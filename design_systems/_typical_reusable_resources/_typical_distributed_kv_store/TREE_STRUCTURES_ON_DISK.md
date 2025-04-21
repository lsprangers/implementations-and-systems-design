# Data Structure and Implementation
Why do we need any of this?
- Durability is important
- Keeping an entire hash table in RAM is sometimes impossible, even if it's split amongst multiple machines
    - We could keep splitting and splitting and splitting, but if one of the machines goes down, we lose that data
- If we wanted to write these hash tables to disk we could just write it in whatever order, but then lookup would be $O(n)$ full scans over all of our disks, which is very slow...so we want to sort things to make it $O(log n)$
- Another example is a Relational Database Index - like a hash index. They basically duplicate the entire columns data footprint (pointers only) so they're huge, and most databases ensure durability, so we can’t store in RAM
    - Need to use a different indexing structure 
        - Can split entire hash index into smaller segments of indexes
        - Store these indexes on disk
        - Retrieve only the indexes and keys we need from disk into memory
        - How to find indexes? Iterate over them all?
        - This is bad – we can use a B-Tree structure so that retrieval is O(log(n))
    - Ultimately this is cascading indexes, where we store larger ranges on disk and recursively use binary search over all of the cascading indexes to find keys/memory locations
    - This looks like a tree data structure
    - This data structure is known as  B-Tree Data Structure
        - Self-balancing tree data structure
            - This means as new nodes are added or deleted then tree modifies it’s structure to keep height small
        - Allows searches, insertions, and deletions in log time 
        - B-Tree helps to break down data into fixed-size blocks called pages where each page is usually several KB
        - Page contains data or references to data
        - One page is designated as root, and then we cascade down
- Between writing Distributed KV Stores to disk, or using a Hash Index in a relational database, and many other examples we don't list here, there are reasons to use Tree Structures on Disk

So long story short, ***disk writes of sorted data are useful in multiple scenarios, and the first place we see them is Distributed KV Stores***

The two most common are B+ Trees, and LSMTree's

# Structures
The [Disk Based Folder](../../_reference%20material/databases%20&%20storage/Disk%20Based/) has more thorough discussions of all of these 

## B Tree

### How They Work:
- B-Trees are self-balancing tree data structures that split data into fixed-size blocks (pages) stored on disk.
- Each page contains keys and pointers to child pages, allowing efficient traversal
- Operations like search, insert, and delete are performed in $ O(\log(n)) $ time
- B-Trees are designed to minimize disk I/O by keeping the tree height small and reading/writing data in large blocks (pages)
    - Self-balancing so that it's not a long tail down, similar to UnionFind, Red-Black Tree, AVL Tree, and other self-balancing data structures
    - These take a small cost during writes to uphold properties, but allow efficient reads

### Why They Ensure Durability:
- Data is written directly to disk in a structured manner.
- Changes (e.g., inserts, updates, deletes) are propagated to disk immediately or in batches.
- B-Trees are commonly used in databases like MySQL (InnoDB) and PostgreSQL for indexing.

### Use Case:
- Best for workloads with a mix of reads and writes
    - Writes are still fast even with self balancing
- Suitable for systems where data is frequently updated and needs to be immediately durable

---

## LSM Trees (Log-Structured Merge Trees)

### How They Work:
- LSM Trees are optimized for write-heavy workloads by batching writes in memory and periodically merging them into sorted files on disk
- Data is first written to an in-memory structure (e.g., a sorted tree or skip list) and then flushed to disk in sorted order
- Disk writes are append-only, which reduces random I/O and improves write performance
- Reads may require merging data from multiple on-disk files, but this is mitigated by maintaining Bloom filters and indexes

### Why They Ensure Durability:
- Writes are first logged in a **Write-Ahead Log (WAL)** to ensure durability in case of crashes
- Data is periodically flushed to disk in immutable segments, ensuring no data is lost
- Compaction processes merge and reorganize on-disk data to optimize reads and reclaim space

### Use Case:
- Best for write-heavy workloads where data is frequently appended or updated
- Used in modern KV stores like **Cassandra**, **RocksDB**, **LevelDB**, and **HBase**

---

## Comparison: B-Trees vs. LSM Trees

| **Aspect**              | **B-Trees**                              | **LSM Trees**                              |
|-------------------------|-------------------------------------------|--------------------------------------------|
| **Write Performance**   | Moderate (writes directly to disk).       | High (writes batched in memory).           |
| **Read Performance**    | High (data is well-indexed).              | Moderate (may require merging on-disk files). |
| **Disk I/O**            | Random I/O (writes to specific pages).    | Sequential I/O (append-only writes).       |
| **Durability**          | Immediate (writes directly to disk).      | Ensured via WAL and periodic flushes.      |
| **Use Case**            | Balanced read/write workloads.            | Write-heavy workloads.                     |

---

## Other Data Structures for Durable KV Stores

While B-Trees and LSM Trees are the most common, other data structures are also used depending on the workload:

1. **Hash Tables**:
   - Used for in-memory KV stores like **Redis**
   - Not inherently durable but can be made durable with periodic snapshots or append-only logs

2. **Trie (Prefix Tree)**:
   - Used in systems like **TrieDB** for efficient prefix-based lookups
   - Less common for general-purpose KV stores

3. **Skip Lists**:
   - Used in **MemTable** implementations for LSM Trees (e.g., RocksDB)
   - Efficient for in-memory operations but not directly used for durability

4. **Write-Ahead Logs (WAL)**:
   - Not a data structure but a mechanism to ensure durability
   - Used alongside other structures (e.g., LSM Trees) to log changes before they are applied to the main data structure

---

## Conclusion

**B-Trees** and **LSM Trees** are the two most common data structures for ensuring durability in KV stores because they are optimized for disk-based storage and provide efficient read/write operations. However, the choice between them depends on the workload:
- Use **B-Trees** for balanced read/write workloads
- Use **LSM Trees** for write-heavy workloads
