# problem 1443

from typing import List
from collections import defaultdict

class Solution:
    def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        # bidirectional tree
        tree = defaultdict(list)
        for s,e in edges:
            tree[s].append(e)
            tree[e].append(s)

        def dfs(cur_node, parent_node):
            res = 0
            for neighbor in tree[cur_node]:
                if neighbor != parent_node:
                    res += dfs(neighbor, cur_node)
            
            # case1, res != 0, this means we have found some apples down the tree and we will have to go
            #   back up to the parent node
            # case2, hasApple[node]==True, this means the current node has a apple on it
            if res or hasApple[cur_node]:
                return res + 2
            
            return res
        
        # Following the dfs, you can see that when we coming back to node 0, if there are some apples in the tree,
        # We added an extra 2 to the result, so we need to -2 here.
        # In case there is no apples in this tree, dfs will return 0, but we can't return -2, so just return 0
        return max(dfs(0,-1)-2, 0)