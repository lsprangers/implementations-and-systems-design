# Graph Processing
Throughout work we've found the need to process humongous graph datasets, and to serve them

Processing large datasets based on graph structures is usually done in a distributed manner using GraphX (Scala) or GraphFrames (PySpark) on Spark....at least from what we've seen

Serving and making this data available to API's usually means efficiently storing this on a backend app database and exposing to API's. GraphQL is useful for exploration and allowing for API based querying, but isn't necessarily real time where other options like Neo4J or TigerGraph allow for real-time graph querying, but aren't useful for analytics.

It's the common OLAP vs OLTP discussion, but it's just on data comprised of nodes and edges!

# Graph Data Processing and Analytics

## Pregel
[Pregel](../z_arxiv_papers/Pregel%20MarkedUp.pdf) is an initial graph processing system built by Google that proposes a way to do Graph Traversals based on message-passing protocols

The entirety of an algorithm is based on ***super-steps***, where programs are expressed as a sequence of iterations, in each of which a vertex can receive messages sent in the previous iteration, send messages to other vertices, and modify its own state and that of its outgoing edges or mutate graph topology

This basically comes down to BFS iterations, where in each step we expand another step outwards (except in a graph we just follow another hop on the graph), and update nodes there, and if we need to continue updating nodes we push into the queue for next hop

This allows us to run many programs such as Connected Components, Transitive Closure (all reachable nodes from any node), and Shortest Path in an efficient and distributed way

## GraphX
GraphX is a Pregel-like implementation that runs on Spark via Scala SDKs

We ended up using this for most of our applications because of the native integration with Spark and by proxy Databricks. This gives us a highly available, distributed, parallel, graph computation engine

### Triplets
Triplets are the main data structures in GraphX, and can help us define Edges

`EdgeTriplet[VD, ED]`

Which helps to implement `(srcId, srcAttr) --[attr]--> (dstId, dstAttr)`, which can ultimately give us contextual access to vertex and edge information during traversal

It contains:
- ***srcId***: ID of the source vertex
- ***dstId***: ID of the destination vertex
- ***srcAttr***: attribute of the source vertex
- ***dstAttr***: attribute of the destination vertex
- ***attr***: the edge’s attribute

```
// For Graph[String, String]
graph.triplets.map { triplet =>
  s"${triplet.srcAttr} --[${triplet.attr}]--> ${triplet.dstAttr}"
}
```

An example would be only processing source nodes in which we haven't seen all of the destination nodes - this would constitute updates / inserts in the graph we haven't processed
```
graph.pregel(initialMsg)(
  vprog = (id, attr, msg) => attr ++ msg,
  sendMsg = triplet => {
    if (!triplet.dstAttr.containsAll(triplet.srcAttr)) {
      Iterator((triplet.dstId, triplet.srcAttr))
    } else Iterator.empty
  },
  mergeMsg = (a, b) => a ++ b
)
```

## Flink Gelly
TODO?

## Transitive Closure Example
Transitive Closure helps us to identify all transitive paths between records that are connected from other nodes

A &rarr; B &rarr; C &rarr; D

Using a few different methods we can calculate:
- A eventually will connect to D
- A connects to D at exactly the 3rd hop
- A connects to D via B connecting to C

# Graph Serving

## GraphQL

## Neo4J