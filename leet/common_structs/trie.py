class TrieNode:
    def __init__(self):
        self.links = [None] * 26
        self.is_end = True
    
    def contains_next(self, char: str) -> bool:
        return(
            self.links[ord(char) - ord('a')] is not None
        )
    
    def put_link(self, char: str, node: 'TrieNode') -> None:
        self.links[ord(char) - ord('a')] = node
    
    def get_char(self, char: str) -> 'TrieNode':
        return(self.links[ord(char) - ord('a')])
    
    def set_end(self) -> None:
        self.is_end = True

    def is_end(self) -> bool:
        return self.is_end    
    

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if not node.contains_key(char):
                node.put(char, TrieNode())
            node = node.get(char)
        node.set_end()

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if not node.contains_key(char):
                return(False)
            node = node.get(char)
        return(True)

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if not node.contains_key(char):
                return(False)
            node = node.get(char)
        return(node.is_end())    