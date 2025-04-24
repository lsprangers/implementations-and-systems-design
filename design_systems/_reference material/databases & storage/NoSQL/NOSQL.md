# General NoSQL Databases

TODO: This is all ChatGPT garbage

NoSQL databases are a class of databases designed to handle large-scale, distributed, and unstructured or semi-structured data. Unlike traditional relational databases, NoSQL databases provide flexibility in data modeling and are optimized for specific use cases such as high availability, scalability, and low-latency queries.

---

## Table of Contents
- [Introduction to NoSQL Databases](#introduction-to-nosql-databases)
- [CAP Theorem and Tradeoffs](#cap-theorem-and-tradeoffs)
  - [Consistency](#consistency)
  - [Availability](#availability)
  - [Partition Tolerance](#partition-tolerance)
  - [Tradeoffs in NoSQL Databases](#tradeoffs-in-nosql-databases)
- [Data Models and Partition Keys](#data-models-and-partition-keys)
  - [Key-Value Stores](#key-value-stores)
  - [Document Stores](#document-stores)
  - [Column-Family Stores](#column-family-stores)
  - [Graph Databases](#graph-databases)
- [Efficiency Gains Using Indexing and Query Patterns](#efficiency-gains-using-indexing-and-query-patterns)
  - [Indexing in NoSQL Databases](#indexing-in-nosql-databases)
  - [Query Patterns](#query-patterns)
- [Conclusion](#conclusion)

---

## Introduction to NoSQL Databases

NoSQL databases are designed to address the limitations of traditional relational databases in handling:
- High-velocity data.
- Large-scale distributed systems.
- Flexible and dynamic schemas.

Common types of NoSQL databases include:
1. **Key-Value Stores** (e.g., Redis, DynamoDB).
2. **Document Stores** (e.g., MongoDB, Couchbase).
3. **Column-Family Stores** (e.g., Cassandra, HBase).
4. **Graph Databases** (e.g., Neo4j, Amazon Neptune).

---

## CAP Theorem and Tradeoffs

The **CAP theorem** states that in a distributed system, it is impossible to simultaneously guarantee all three of the following:
1. **Consistency (C)**: Every read receives the most recent write or an error.
2. **Availability (A)**: Every request receives a response, even if it is not the most recent write.
3. **Partition Tolerance (P)**: The system continues to operate despite network partitions.

### Consistency
- Ensures that all nodes in the system have the same data at any given time.
- Prioritizing consistency may result in slower writes or reduced availability during network partitions.

### Availability
- Ensures that the system remains operational and responsive, even if some nodes are unreachable.
- Prioritizing availability may result in stale or inconsistent data being served.

### Partition Tolerance
- Ensures that the system can handle network failures or partitions without crashing.
- Partition tolerance is a fundamental requirement for distributed systems.

### Tradeoffs in NoSQL Databases
NoSQL databases make tradeoffs between consistency and availability, depending on the use case:
- **CP Systems**: Prioritize consistency over availability (e.g., HBase, MongoDB in strong consistency mode).
- **AP Systems**: Prioritize availability over consistency (e.g., DynamoDB, Cassandra).
- **CA Systems**: These are not possible in distributed systems with partitions.

---

## Data Models and Partition Keys

NoSQL databases use flexible data models to optimize for scalability and performance. Partition keys play a critical role in distributing data across nodes in a cluster, because they allow us to shard data out into separate nodes which increases throughput and scalability

### Key-Value Stores
- **Data Model**: Simple key-value pairs
- **Partition Key**: The key is used to determine the partition where the value is stored
- **Use Case**: Caching, session management, real-time analytics
- **Example**:
```json
{
"user123": {
    "name": "John Doe",
    "age": 30
}
}
```


### Document Stores
- **Data Model:** JSON-like documents with nested structures.
- **Partition Key:** A field in the document (e.g., user_id) is used to distribute data.
- **Use Case:** Content management systems, e-commerce catalogs.
```
{
  "user_id": "user123",
  "name": "John Doe",
  "orders": [
    {"order_id": "order1", "amount": 100},
    {"order_id": "order2", "amount": 200}
  ]
}
```

### Column Family Stores
- **Data Model:** Rows and columns grouped into column families.
- **Partition Key:** A primary key (or composite key) determines the partition.
- **Use Case:** Time-series data, logs, analytics.
- **Example:**
    - Row Key: user123
    - Column Family: PersonalInfo
        - Name: John Doe
        - Age: 30
    - Column Family: Orders
        - Order1: 100
        - Order2: 200

### Graph Databases
- **Data Model:** Nodes and edges representing entities and relationships.
- **Partition Key:** Nodes are partitioned based on their IDs or relationships.
- **Use Case:** Social networks, recommendation systems, fraud detection.
- **Example:** 
```
Node: User123
  - Name: John Doe
Edge: User123 -> User456 (Friend)
```

## Efficiency Gains Using Indexing and Query Patterns

Efficient data retrieval is critical in NoSQL databases, especially when dealing with large-scale distributed systems. Indexing and query patterns play a key role in optimizing performance.

---

### Indexing in NoSQL Databases

Indexes allow NoSQL databases to quickly locate data without scanning the entire dataset. Different types of indexes are used depending on the database and query requirements:

1. **Primary Index**:
   - Automatically created on the partition key
   - Used to locate data within a specific partition
   - Example (DynamoDB):
     - Partition Key: `user_id`
     - Query: Retrieve all data for `user_id = "user123"`

2. **Secondary Index**:
   - Created on non-primary fields to support additional query patterns
   - Useful for querying attributes other than the partition key
   - Example (MongoDB):
     - Index on `email` field to allow queries like:
       ```json
       { "email": "john.doe@example.com" }
       ```

TODO: Secondary indexes are basically points but cause bloat sometimes, espeically if we force them across partitions

3. **Composite Index**:
   - Combines multiple fields into a single index to support complex queries.
   - Example (Cassandra):
     - Composite key: `user_id` and `timestamp` to query time-series data efficiently.

4. **Global Secondary Index (GSI)**:
   - Used in distributed databases like DynamoDB to query across partitions.
   - Example:
     - Index on `order_date` to retrieve all orders placed on a specific date.

5. **Full-Text Index**:
   - Used for text search in document stores like Elasticsearch or MongoDB.
   - Example:
     - Index on `description` field to support keyword searches.

---

### Query Patterns

Efficient query patterns are essential for optimizing performance in NoSQL databases. Here are some common patterns:

1. **Query by Partition Key**:
   - Always include the partition key in queries to avoid full table scans.
   - Example (DynamoDB):
     ```sql
     SELECT * FROM Users WHERE user_id = 'user123';
     ```

2. **Denormalization**:
   - Store redundant data to optimize for read performance.
   - Example:
     - Embed user orders directly in the user document:
       ```json
       {
         "user_id": "user123",
         "name": "John Doe",
         "orders": [
           { "order_id": "order1", "amount": 100 },
           { "order_id": "order2", "amount": 200 }
         ]
       }
       ```

3. **Materialized Views**:
   - Precompute and store query results for faster access.
   - Example:
     - A view that aggregates total sales per user:
       ```json
       {
         "user_id": "user123",
         "total_sales": 300
       }
       ```

4. **Time-Series Queries**:
   - Use composite keys to efficiently query time-series data.
   - Example (Cassandra):
     - Partition Key: `user_id`
     - Clustering Key: `timestamp`
     - Query: Retrieve all events for a user within a specific time range:
       ```sql
       SELECT * FROM Events WHERE user_id = 'user123' AND timestamp > '2023-01-01';
       ```

5. **Query Optimization**:
   - Design queries to minimize the number of partitions accessed.
   - Avoid operations like `SCAN` or `FILTER` that require processing large amounts of data.

---

### Best Practices for Indexing and Query Patterns

1. **Design for Read Patterns**:
   - NoSQL databases are optimized for specific query patterns. Design your schema to match the most common queries.

2. **Minimize Index Overhead**:
   - Indexes improve read performance but can slow down writes. Only create indexes for fields that are frequently queried.

3. **Use Partition Keys Effectively**:
   - Choose partition keys that evenly distribute data across nodes to avoid hotspots.

4. **Monitor Query Performance**:
   - Use database monitoring tools to identify slow queries and optimize them.

---

By leveraging indexing and designing efficient query patterns, NoSQL databases can achieve high performance and scalability, even with large datasets and distributed architectures.