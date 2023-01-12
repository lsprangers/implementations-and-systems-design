# problem 1519
from typing import List 
from collections import defaultdict, Counter

class Solution:
    def countSubTrees(self, n: int, edges: List[List[int]], labels: str) -> List[int]:
        tree = defaultdict(list)
        for s,e in edges:
            tree[s].append(e)
            # tree[e].append(s)
        
        res = [0] * n
        
        def dfs(cur_node):
            # keep res as a nonlocal variable - we want it to be te same
            #   across recursive calls
            nonlocal res 

            # keep a counter to count everything "below" this in the tree
            count = Counter()

            for neighbor in tree[cur_node]:
                count += dfs(neighbor)

            count[labels[cur_node]] += 1
            res[cur_node] = count[labels[cur_node]]

            return count

        dfs(0)
        return res