# Relational SQL
SQL Databases are used for relational data modeling, and allow us to have typical dimensional-fact, aka dim-fact, tables and models

They also allow us to do things like joining, search / lookup, and CRUD operations on our data after defining the model

## ACID
The [Isolation Levels Documentation](../../../_typical_reusable_resources/_typical_distributed_kv_store/ISOLATION_LEVELS.md) in the KV Store Implementaion discussion talks about all of the different Isolation Levels of databases that defines how transactions and "dirty reads" happen in KV Stores, especially distributed ones

These can also happen in relational databases, but typically they aren't an issue because a majority of SQL Databases are ACID Compliant

- Provide ACID atomic, consistent, isolated, and durable properties
    - ***Atomic*** – Transaction is a singular unit, so all of it will complete or it fails
    - ***Consistent*** – At any given time the database is in a consistent state, and anyone who reads a record gets the same record back
    - ***Isolation*** – In case of multiple transactions running concurrently, they shouldn’t affect each other…the final state should be the same as if they were executed concurrently
    - ***Durable*** – Transaction should live in database even in case of system failures
- Impedance Mismatch – Is the idea that relational databases can’t store the data as it is in memory, so a map or a list can’t be stored directly as it is in a relational database

Most databases with transactions will ensure serialization so that there aren't even phantom reads that are possible

- Dirty Reads:
    - Occurs when a transaction reads uncommitted changes made by another transaction.
    - ACID Compliance:
        - Dirty reads are not allowed in ACID-compliant databases when using isolation levels Read Committed or higher

- Non-Repeatable Reads:
    - Occurs when a transaction reads the same row twice and gets different results because another transaction modified the row in between.
    - ACID Compliance:
        - Non-repeatable reads are not allowed in ACID-compliant databases when using isolation levels Repeatable Read or higher

- Phantom Reads:
    - Occurs when a transaction reads a set of rows that match a condition, but another transaction inserts or deletes rows that match the same condition, causing the first transaction to see different results on subsequent reads.
    - ACID Compliance:
        - Phantom reads are not allowed in ACID-compliant databases when using the Serializable isolation level

## Replication - Consistency, Durability, Write 
The [Replication Documentation](../../../_typical_reusable_resources/_typical_distributed_kv_store/REPLICATION.md) in the KV Store Implementaion discusses a lot of the nuances around replication, especially sync vs async, and how it affects Consistency, Durability guarantees, and Read vs Write times

# Architecture