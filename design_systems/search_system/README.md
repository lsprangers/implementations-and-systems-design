# Table of Contents
- [Search Systems](#search-systems)
- [Inverted Index](#inverted-indexes)
- [Ranking & Comparing](RANKING.md#ranking-first-pass-at)
    - [Count Based Heuristics](RANKING.md#count-based-heuristics)
    - [Probabilistic Models](RANKING.md#probabilistic-models)
    - [Graph](RANKING.md#graph)
    - [BERT](RANKING.md#bert)
- [Context & Embeddings](#context)
    - [Embeddings](../../other_concepts/EMBEDDINGS.md#embeddings)
        - [Collaborative Filtering](../../other_concepts/EMBEDDINGS.md#collaborative-filtering)
        - [Matrix Factorization](../../other_concepts/EMBEDDINGS.md#matrix-factorization)
        - [Attention](../../other_concepts/EMBEDDINGS.md#attention)
- [Real Time Serving Systems](#real-time-serving-systems)
    - [Candidate Generation](#candidate-generation)
    - [Ranking](#ranking)
    - [Serving](#serving)
- [Youtube DNN Paper + Discussion](#youtube-dnn-system)
    - [Candidate Generation](#candidate-generation-1)
        - [User Embeddings](#user-embeddings-1)
        - [KNN Neighbors](#knn)
    - [Ranking](#ranking-1)
        - [Learn To Rank](#learn-to-rank)

# Search Systems
Search Systems (also Recommendation systems since we recommend something back) are used for finding relevant content based on a query

- "What day is it today"
- "Michael Jordan Dunk"
- etc...

All of these things are queries and we'd expect different content to be returned

Youtube will typically return Videos, Google will return almost any content type, App Store would return applications, and Facebook might return posts / users (friends)

## Terminology
- An ***item*** is generally the thing we'd want to recommend
- A ***user*** uses items and can be recommended items
    - Users have a history of item usage
- A ***query*** comes in at some time, usually from a user, and we would recommend items for that query
    - A query can be considered the general context / information a system uses to make recommendations
- Recommendation Scenarios:
    - *Log-On:* When a user logs on and the system will recommend items to them
    - *Search:* When a user queries for a specific item and we return items based on that query
    - For each of the types above we will need to *project* the query context into some sort of *embedding space* and ultimately return some sort of Top K item results to the user
- General architecture:
    - ***Candidate Generation:*** is where we efficiently distill the entire corpus of items down to a manageable size - this typically has high precision where anything we return is good, but we might have missed some good items in the process
    - ***Scoring:*** is where we take the distilled space and run large scale heavy computations against the query / user context to decide on the best videos. This step typically has high recall where we will ensure we get every possible candidate the user might find interesting
    - ***Re-Ranking:*** Takes into account user history and metadata to remove items from scoring that other systems / history have explicitly stated aren't relevant. We wouldn't want to do this in scoring since it would cause lag and we want to keep that system generally abstract. This step also helps us to do experimentation, freshness, and fairness of item retrieval.

# History
Over time recommendation / search systems have gone through a lot of changes 
- At first we used inverted indexes for text based lookups of documents which would allow things like webpage lookup on google
- Over time 
    - Recommendation systems started to span multiple content types, from videos to other users to generic multimedia, and the systems had to keep up
    - Companies started to have humongous web scale for items like Amazon, Google, and Facebook
    - These evolutions led to new search systems that had multiple stages across various content types ***which led systems to converge on Candidate Generation and Scoring over projected embeddings***

Search has started to move away from returning items to returning summaries and question answering live through "GenAI", but in reality this is mostly still based on Transformer models and NLP tasks where we surround it with new context / query information


## Inverted Indexes
[Inverted Indexes](INVERTED_INDEX.md) have been around for a long time, and they built the original search systems we think of. When you hear of "Google indexed my page" or "Google crawled my page" it is referring to a system similar to this

There's many still used today, but for the most part systems require utilizing context, user features / demographics, and many other inputs to help design Search and Recommendation Systems

# Scalable, Real Serving Systems
We'll walk through how serving systems would be architected in the current world

## Candidate Generation
The [Candidate Generation](./CANDIDATE_GENERATION.md) sub-document covers all main areas of candidate generation phase, but in most cases we'll basically be creating the user-item matrices as a batch at some specific time, and then updating it at some cadence as users interact with our service

If we choose to simply use filtering methods then all of the updates and batch creation can be done offline, and if we truly want our recommendations to be as up-to-date as possible we'd have to rerun the WALS update of user-item engagement each time a user uses our service

If we choose DNN, the DNN needs to be ran each time for a specific user to get the output Candidate Generation which leads us into ML Engineering Inference API's

### Embedding Space Updates
How do we update our embedding space as users use our services?

We would need to capture the user click information as it's happening, and stream that data into a user database or an analytical data warehouse 

### User-Item Matrix Updates
Once the data is in some sort of processing engine, we'd need to update the specific pointed row-column $r_{ij}$ corresponding to user I on item J. This might be incrementing some usage statistic, upadting metrics on the fly, or something else. This can be apart of *Feature Engineering* pipelines that run on streaming data

The toughest part will be recomputing the user-item embeddings using WALS or SGD, as we'd have to clone the matrix somewhere else or pause updates on it as we created our new latent matrices $U$ and $V$ during [Matrix Factorization](./CANDIDATE_GENERATION.md#matrix-factorization) 

Then as the user returns, we'd have updated embeddings to serve them with

### DNN Updates
The DNN needs to be ran each time for a specific user to get the output Candidate Generation, and updating the model parameters each time wouldn't be smart so DNN gets retrained as our training data drifts from our User-Item data

The data drift detection can be a separate background feature pipeline in our processing engine, and once there's a significant enough change we can schedule a new model to be trained for inference

## Ranking
## Serving

# Youtube DNN System
[Paper Link](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)

So what could Youtube use in it's recommender system?

## Candidate Generation
- Candidate Generation is around finding a manageable set of videos to compare to an incoming user Query in a very short timeframe 
    - Youtube paper mentions "taking recent user history into context and outputting a small sample of videos (100's) from our total corpus"
- It will have high precision meaning anything it generates is most likely relevant to the user
    - It "provides broad personalization via collaborative filtering"
- The actual DNN is a non-linear generalization of matrix factorization techniques
    - This basically means they used to use matrix factorization techniques, and the DNN means to mimic that, but DNN's are more flexible (non-linear)
    - They mention the CGeneration task is extreme classification
### User Embeddings
### KNN

## Ranking
- Ranking will take the output of Candidate Generation, which is high precision, and will create a fine-level representation for the user 
- Ranking will have high recall to ensure out of the videos Candidate Generation finds, Ranking doesn't leave anything behind
    - It ranks these videos with a rich set of Query (User) - Document (Video) Features
### Learn To Rank
This would be a good first thought - we could basically have user context as input along with some previous history, and then we could rank potential videos that get passed through from Candidate Generation