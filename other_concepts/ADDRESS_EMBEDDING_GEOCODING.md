# Addresses
I thought addresses deserved a separate document because they're so unnecessarily evil. The main "embeddings" for string addresses are just lat/long coordinates since they literally represent the numeric description of the text, and they allow for good distance calculations

The tough part is getting `100 Main St, Springfield, IL` and `100 Main St, Springfieid, IA` to equate to two totally different lat/longs even though the string differs only by a single letter

Clean data and text is by far the most useful part of all of this, and if your input data is clean you can get fairly creative with what you do to convert it into a lat/long on a map

It's difficult to just [Embed](./EMBEDDINGS.md) the text with any old model like [Word2Vec](./EMBEDDINGS.md#word2vec) without using some serious preprocessing and context rules...chunking out the tokens is useful, and can get [decent results](https://datascience.stackexchange.com/a/108692), but ideally focusing on GeoCoding to Latitude / Longitude pairs is the better route. However, GeoCoding is tough and relies on expensive outside services, and so many groups for things like Record Linkage and Entity Resolution might just use Text Embeddings

## Data Parsing
If your data is not clean then you'll need some sort of cleaning / standardization on top of it

[LibPostal](https://github.com/openvenues/libpostal) should forever and always be the first start for anywhere because of their contributions to the open source community. You can use C++, Java, Python, Ruby, Go, NodeJS and many others on Linux and Windows OS's

- Basic standardization completed to clean up the text is useful before sending to LibPostal - most of the time we need to do things like:
    - ***Transliteration*** to bring our data from any alphabet type to Latin
        - ***I get that sometimes other languages are drastically more useful, if we're doing address encoding in China we certainly shouldn't convert to Latin alphabet
    - ***Character Cleaning*** to remove non-alphanumeric items
    - ***Character Replacement*** using standard sets of dictionaries to correct common errors

Once all of this is completed (or not) sending it through to LibPostal should give you a parsed, potentially normalized, representation of your address. If you send `100 Main Stz.` through, `Stz.` might come as `road_type` instead of `St.` but it's all a case by case basis

## Similarities
Once we've parsed and normalized our data we can upload it to a service to get compared to other locations we know of

## Hierarchical Boundaries
Can start bounding addresses by the Continent, Country, Region, State, City, etc... hierarchy for each region of the world. Not all regions have States, and so typically there are known hierarchies based on the Country and Region we'd lookup to start filtering downwards

## Address Number Lookup
`from_node` is typically the starting address of a street, and `to_node` is typically the end. If our `Main Street` has street numbers from $[100, 400]$ and we lookup $200$, then it will go $1/3$ of the way "down the street", known as ***interpolating*** from our starting node to get that lat/long

## Bounding Box Search Space
We can reduce the search space for something like `100 Main St` if we have User context around what the user is looking for. For example if they're in Iowa instead of Illinois, we can filter down to just Iowa addresses and check there...but this can be tricky because cities like Kansas City are on the border of 2 states, so we can't use hierarchies based on lines drawn, but we'd have to use ***bounding boxes***

## Map Zoom
We can use the User context of the location of the world a user is checking on the UI to reduce the search space (if they're looking at a map of USA they probably don't want EU results)

## GeoIP
We can use User context like their GeoIP to further reduce the search space