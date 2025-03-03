# Search System (Youtube for example)
Search Systems (also Recommendation systems since we recommend something back) are used for finding relevant content based on a query

- "What day is it today"
- "Michael Jordan Dunk"
- etc...

All of these things are queries and we'd expect different content to be returned

Youtube will typically return Videos, Google will return almost any content type, and Facebook might return posts

# Inverted Indexes
[Inverted Indexes](INVERTED_INDEX.md) have been around for a long time, and they built the original search systems we think of. When you hear of "Google indexed my page" or "Google crawled my page" it is referring to a system similar to this

There's many still used today, but for the most part systems require utilizing context, user features / demographics, and many other inputs to help design Search and Recommendation Systems

# Ranking
Once we receive documents back from a query there are times where we'd want to rank them and only return the Top K documents or relevant items

[Ranking](RANKING.md) will essentially open up the search world into context, embeddings, and ways to score text compared to each other. This is typically done with comparing an input Query Q to your set of Documents D, maybe including some Context C

Foreshadowing here in the case of Ranking, even with simple architectures like TF-IDF for scoring, we still have 2 general phases for retrieval which is [Candidate Generation](#candidate-generation) and [Ranking](RANKING.md)...we are skipping over candidate generation for now because scoring is easier to think about

This is a "simple" architecture where we shove everything into BLOB storage, but for TFIDF ranking we'd probably keep it in something like a [Distributed KV Store](/design_systems/_typical_reusable_resources/_typical_distributed_kv_store/README.md) so that we have an "offline" phase for batch indexing, and an "online" phase for applications to return data fast and that would be built over a KV store and not BLOB storage
![General Architecture of TFIDF](./images/inverted_index_tfidf.png)

# Context

## Embeddings

## History

## Attention

# Youtube DNN System
[Paper Link]()

## User Embeddings
## Candidate Generation
## Ranking
## Serving