import hashlib
import bisect
from typing import List, Dict, Any

class ConsistentHashing:
    def __init__(self, nodes: List[str], replicas: int = 3):
        self.replicas = replicas
        self.ring = []
        self.node_map = {}
        for node in nodes:
            self.add_node(node)

    def add_node(self, node: str):
        for i in range(self.replicas):
            key = self.hash(f"{node}:{i}")
            self.ring.append(key)
            self.node_map[key] = node
        self.ring.sort()

    def remove_node(self, node: str):
        for i in range(self.replicas):
            key = self.hash(f"{node}:{i}")
            self.ring.remove(key)
            del self.node_map[key]

    def get_node(self, key: str) -> str:
        if not self.ring:
            return None
        hashed_key = self.hash(key)
        idx = bisect.bisect(self.ring, hashed_key)
        if idx == len(self.ring):
            idx = 0
        return self.node_map[self.ring[idx]]

    @staticmethod
    def hash(key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

class RaftNode:
    # RAFT repo has a lot more than here
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.leader = None
        self.followers = []

    def set_leader(self, leader: str):
        self.leader = leader

    def add_follower(self, follower: str):
        self.followers.append(follower)

    def is_leader(self) -> bool:
        return self.node_id == self.leader

    def handle_request(self, key: str, value: Any = None) -> Any:
        if self.is_leader():
            if value is not None:
                # Handle write request
                return self.write(key, value)
            else:
                # Handle read request
                return self.read(key)
        else:
            # Forward request to leader
            return self.forward_to_leader(key, value)

    def write(self, key: str, value: Any) -> str:
        # Implement write logic
        return f"Key {key} written with value {value}"

    def read(self, key: str) -> Any:
        # Implement read logic
        return f"Value for key {key}"

    def forward_to_leader(self, key: str, value: Any = None) -> Any:
        # Implement forwarding logic
        return f"Forwarded to leader {self.leader}"

class DistributedKVStore:
    def __init__(self, nodes: List[str]):
        self.consistent_hashing = ConsistentHashing(nodes)
        self.raft_nodes = {node: RaftNode(node) for node in nodes}
        self.setup_raft_clusters(nodes)

    def setup_raft_clusters(self, nodes: List[str]):
        for node in nodes:
            leader = node
            followers = [n for n in nodes if n != node]
            self.raft_nodes[node].set_leader(leader)
            for follower in followers:
                self.raft_nodes[node].add_follower(follower)

    def put(self, key: str, value: Any):
        node = self.consistent_hashing.get_node(key)
        return self.raft_nodes[node].handle_request(key, value)

    def get(self, key: str) -> Any:
        node = self.consistent_hashing.get_node(key)
        return self.raft_nodes[node].handle_request(key)

# Example usage
nodes = ["node1", "node2", "node3"]
kv_store = DistributedKVStore(nodes)

print(kv_store.put("key1", "value1"))
print(kv_store.get("key1"))