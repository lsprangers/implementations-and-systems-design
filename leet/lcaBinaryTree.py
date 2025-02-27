# 1650
"""
# Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
"""

class Solution:
    def __init__(self):
        self.seen = set()

    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        qu = deque([])
        qu.appendleft(p)
        qu.appendleft(q)
        while qu:
            qu_size = len(qu)
            for _ in range(qu_size):
                top = qu.pop()
                if top == None:
                    continue
                    
                # if we've seen this from another traversal, it must be LCA
                if top.val in self.seen:
                    return(top)

                self.seen.add(top.val)
                qu.appendleft(top.parent)
        
        return(p)