# Redis PubSub
Redis, being a distributed KV cache, doesn't sound like the system to use for [PubSub](../../messaging/PubSub/PubSub.md), but Redis has a lot of expertise in this area

## Non Functional Requirements
Altogether Redis PubSub does not store messages long term, and doesn't help offline subscribers get caught up - it's really good at flinging messages around in real time in a "stupid" way meaning it ignores offsets and state

### Storage
Redis doesn't store messages to disk long term, and doesn't usually allow for sending messages from an offset

### Scaling
Redis is a single-threaded engine, so scalability usually revolves around horizontal scaling and creating new broker instances to handle increased load

### Latency + Acknowledgement
Typically ***<1ms***, but the PUBLISH command is synchronous meaning it doesn't respond to clients until all Consumers have received the message. Redis brokers simply push out the messages to Consumers without explicit need for any sort of acknowledgements


