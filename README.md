# This Repo
This entire repo is just a dump of a bunch of things I've read and wanted to write down somewhere other than a MS Word file - as many reference links as I can remember are at the bottom

## References
All of these websites are used throughout this, and most of my notes come from these places:
- [Leetcode](https://leetcode.com) is a great site that helps with DSA and Systems Design, and most of the DSA stuff comes from here
- [3Blue1Brown](https://www.3blue1brown.com/) saved me in college, helped me through Linear Algebra and Probability, and is used a lot in DNN discussions
    - The [Attention](https://www.youtube.com/watch?v=eMlx5fFNoYc&vl=en) video is the clearest representation of this I've ever seen
- [Jay Alammars Visual Transformer Paper](https://jalammar.github.io/illustrated-transformer/) is also a huge help in Attention and Transformers
- [Google ML Rec Systems](https://developers.google.com/machine-learning/recommendation) was very useful for everything search, rec systems, and embeddings
- [Hello Interview](https://www.hellointerview.com) is great for Systems Design Interviews
- [Programiz](https://www.programiz.com) is great for implementations of DSA and their complexities

## Design Systems
I forgot a lot of this since school, and had to do this in some of my more technical interviews

Building specific systems based utilizing [Typical Resources](#typical-resources)

Some examples include URL shortener, Youtube, Youtube Search, Top K Heavy Hitters, etc...

To implement all of these things, we need to know about Databases & Storage, Messaging, Calculations and Timings, and other "things" which we make notes about inside

### Databases and Storage
Discusses different SQL and NoSQL architectures / products including Relational Databases, NoSQL KV storage, Blob Storage, and Data Warehousing Solutions

### Messaging
Discuss how to send messages between services via Queues, Brokers, and PubSub systems

### Typical Resources
These are discussions and implementations of "typical" resources like distributed KV store, front end load balancer + metadata, distributed queue, etc...

I actually write out the K8's + Terraform for all of these because it's helpful to see how easy it is to create a lot of these systems natively - a good production system would have many tweaks

## DSA
Just general Data Structures and Algorithms and their implementations, time /space complexities, and general use cases

## Other Concepts
These are generic concepts that don't fit into the other buckets - things like Embeddings, Parallel Training designs, Pregel Graph Traversals, etc... Which I mention, and are useful, but don't belong in DSA or Sys Design implementations

## Implementations
This is a folder of generic implementations of things like
- "A Key Value store with transactions"
- "A balanced binary search tree from scratch"
- "ML feature store architecture diagram"

## Leet
This is just a dump of leetcode problems and comments
