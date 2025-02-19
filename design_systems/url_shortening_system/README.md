# System
A URL shortening system is one where you give some URL and it returns a shortened version of it!
It's pretty straightforward, except the shortened URL typically needs to be kept alive for some period of time
Altogether it's basically another layer in DNS 

### Functional requirements
- Availabile: We can always give a shortened URL, and anyone can find the shortened URL for their origin 
- *Consistent: We don't want multiple short URL's to point to different original URL's, but the actual level of consistency depends
- Scalable: Same thing as always - in this case we'll choose higher consistency over higher availability, because let's say someone told us that's the design choice...we can design both ways

### Non-functional requirements
- URLs should have only alphanumeric components
- URLs should have a known maximum length, say 20 characters 
- We do not make choices around the actual websites it points to (could be nefarious websites)
- There's a typical expiration time of the shortened URL, say a month, where after the client would need to re-register a URL for shorterning

### Key actors
- Users will call this website to get a registered shortened URL for their input URL
- Clients will call this website as a step in DNS resolution
    - `shorturl.com --resolves--> superlongandtediousurl.com`

### Key components
- 

### High-level architecture

How these key components are connected with each other.