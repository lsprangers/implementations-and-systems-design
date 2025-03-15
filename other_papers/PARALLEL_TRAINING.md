From our embeddings in search systems paper we say `DistBelief` mentioned in the 2013 Word2Vec paper

## DistBelief
- Sounds like they update small portions of the model on many different servers, and send all of the updates back to a centralized server which then will push weight updates back through training servers
- Mini-Batch gradient descent with adaptive learning rates via `AdaGrad`