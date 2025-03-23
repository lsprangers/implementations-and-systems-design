# PubSub
[PubSub](../../design_systems/_typical_reusable_resources/_typical_distributed_pubsub/README.md) is the architecture based on Message Channels, where Publishers post messages to the Channel, and Subscribers subscribe to channels to get Messages pushed to them

We design an entire [PubSub](../../design_systems/_typical_reusable_resources/_typical_distributed_pubsub/README.md) System from scratch in this folder

The PubSub broker in the middle takes care of receiving and sending messages

PubSub brokers are different from other message brokers like [Kafka](../Kafka%20Broker/KAFKA.md) by how Consumers interact with the Broker
    - In Kafka the Consumers pull
    - In PubSub the Consumers get pushed to

## PubSub Systems in Products
### Redis PubSub
Redis, being a distributed KV cache, doesn't sound like the system to use for [PubSub](../../databases%20&%20storage/Redis/PUBSUB.md)