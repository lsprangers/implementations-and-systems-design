# Ranking
Ranking queries to items is a fairly large part of a search system, otherwise you're just returning a big list of documents when some might be much more relevant to user

## Uniqueness
One of the most common ranking / scoring methodologies is using the uniqueness of a word - the word "and" is not very unique, and if it shows up in a document we won't really care. Another word like "aardvark" is fairly unique and not used that often, so it would be more unique

## TF-IDF / BM25
TF-IDF means Term Frequency Inverse Document Frequency, and it's a fairly simple scoring mechanism for computing the uniqueness of a Term (word) across Documents 

TF = count of a Term in this specific document
IDF = count of the number of Documents that have that Term in it

| TF         | IDF            | Meaning        | TFIDF
|------------|----------------|----------------|----------------|
| High       | High           | This word is common among all documents, and this document, so it's just a generally common word | Fairly normal score - around mean value | 
High       | Low           | This word is rare throughout the other documents, but comes up a lot in this document, so it must be reasonably relevant for this document| High score |
| Low  | High | This word is a common word, and it's not even showing up much in this document | Low Score |
| Low | Low | This word isn't very common, but it's also not apart of this document, so it's not very relevant | Low but closer to mean | 

![General Architecture of TFIDF](./images/inverted_index_tfidf.png)