from dataclasses import dataclass
from typing import Optional

@dataclass
class Node:
    val: int
    left: Optional['Node'] = None
    right: Optional['Node'] = None

class BSTree:
    def __init__(self):
        self.root = Node(0, None, None)
    
    def search(self, val: int) -> Optional[Node]:
        return self._search(self.root, val)

    def _search(self, node: Node, val: int) -> Optional[Node]:
        if node is None or node.val == val:
            return node
        
        if node.val < val:
            return self.search(node.left)
        
        return self.search(node.right)
    
    def insert(self, val: int):
        if self.root is None:
            self.root = Node(val)
        else:
            self._insert(self.root, val)
    
    def _insert(self, node: Node, val: int) -> None:
        if val < node.val:
            if node.left is None:
                node.left = Node(val)
            else:
                self._insert(node.left, val)
        else:
            if node.right is None:
                node.right = Node(val)
            else:
                self._insert(node.right, val)
    
    def _inorder(self):
        stack = [self.root]
        while stack:
            curr = stack.pop()
            print(curr.val)
            if curr.left:
                stack.append(curr.left)
            if curr.right:
                stack.append(curr.right)

# Example usage
bst = BSTree()
bst.insert(10)
bst.insert(5)
bst.insert(15)
bst.insert(3)
bst.insert(7)

# Search for a value
node = bst.search(7)
if node:
    print(f"Found node with value: {node.val}")
else:
    print("Value not found")
        