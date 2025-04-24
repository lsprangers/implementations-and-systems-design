# Disk Based Storage
Disk based storage underpins almost all "durable storage" systems

Most of the core concepts are the same as volatile memory data structures, except we take these generic data structures and write them to block storage

- Common Disk Based Structures:
    - Tree Based:
        - B-Tree
        - Log Structured Merge (LSM) Tree
    - Tabular Storage
        - Columnar / Column Oriented Storage
        - Row Oriented Storage
    - List Based
        - Linked List
        - Doubly Linked List
        - Skip List
        - Bloom Filter?
    - Hash Table
    - Graph Based
        - Trie
        - Generic Graph
            - Specifically, using Parquet for Pregel based graphs

# Tree Based
Some open source databases use a B-Tree or LSM-Tree as their primary storage engine such as:
- B-Tree
    - MySQL
    - PostgreSQL
    - SQLite
- LSM-Tree
    - Cassandra
    - HBase
    - RocksDB

- At a high level:
    - *BTree's are better for reads and worse for writes* because they are self balancing and require more compute cycles per write, but they're still incredibly fast for disk write structures
    - *LSMTrees are better for writes and worse for reads* because you quickly write the data to MemTable and then forget about it, and background processes ensure it's durable
    - Both are:
        - Durable
        - Consistent (I think)
        - Relatively $O(log n)$ read    
            - Tree lookups can degrade to $O(n)$ if we don't search on the primary sort key 
            - It would force us to do a full scan of the data until we find the key with a specific attribute, and any range queries are guaranteed $O(n)$

## B-Tree
- B-Trees are self-balancing data structures, meaning writes take a bit more time to ensure that reads are always $O (\log n)$
- Leaf nodes hold references to sorted records
- Internal / parent nodes hold references to other nodes
- Writes
    - As new records come in the leaf nodes are updated, and the tree is re-balanced
    - B-Tree good for reads, worse for writes
- Reads
    - Because of the binary search tree structure, reads are very efficient and usually complete in < 5 hops
- B-Tree is a good choice for indexing in relational databases, especially when range queries are required
    - B-Trees are commonly used in relational databases for indexing columns, especially when range queries are required

## LSM-Tree
- Structured as Levels of Sorted Runs
- Sorted Run - Lis of records sorted by key on disk
    - Sorted runs organized into Levels based on max storage of that level
    - Compaction
        - When current level is full / at max, a Sorted Run is moved to the next level and merged with other sorted runs
            - Merged with other runs with overlapping key ranges
- LSM-Tree good for writes, worse for reads
- Reads
    - Reads are done by searching the MemTable, and then searching the SSTables in order from lowest to highest level
    - This means that reads can be $O(n)$ if there are a lot of levels
- Writes
    - Writes are done by writing to the MemTable, and then flushing the MemTable to disk when it is full
    - This means that writes are $O(1)$, but reads can be $O(n)$ if there are a lot of levels

### LSMTree Implementation
We discuss LSMTree implementations in more detail in the [LSMTree Sub-Document](./LSMTREE.md#implementation), but the 3 main components are:
- ***MemTable***
    - Can be any sort of container that supports sorting keys, and scanning retrieval
        - Insert and search can both be O(log n) with a simple Vector
        - RocksDB uses a Skip List which is a Linked List with multiple other links that you can use to skip
        - Skip Lists still have, on average, O(log n) search without O(n) insert which can happen with a Vector if it's full
        - Can't use a Hash Map because scans would be O(n)
    - Since we write to MemTable in O(log n) in RAM, and then write this to disk as SSTable later on, writes are fast
    - Still durable because of WAL
    - We remove need for random disk writes which is how B-Tree would handle it
- ***WAL***
    - On disk backup for MemTable
    - Just an append only log of transactions
    - Uses Sequential I/O and not Random I/O
        - Sequential I/O is faster, but uses up a lot of disk space because each time a record is updated older versions are kept
        - Space Amplification is the amount of disk space per actual database - so if a 1GB Dataset uses 2GB of disk space, it's space amplification is 2
    - Space Amplification is the amount of disk space per actual database - so if a 1GB Dataset uses 2GB of disk space, it's space amplification is 2
- ***SSTable***
    - When a MemTable is flushed to disk, we take that Sorted Run and store it on disk as an SSTable
        - i.e. when we want to write MemTable to disk, we store it as an SSTable
    - SSTables are organized into levels with max capacities of powers
        - 10MB, 100MB, etc...
        - Whenever any level reaches capacity, it's Compacted into next level
        - This means older records will persist on disk in older SSTables until updated
    - Compaction
        - Our garbage collection and data management process of our database
        - Compaction takes SSTable from filled level and merges it with SSTables in next level with overlapping key ranges
            - When files are merged we:
                - Re-sort keys
                - Outdated records are removed
                - New SSTable is created
