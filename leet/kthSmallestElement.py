# 230
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def __init__(self):
        self.num_elements = 0
        self.vals = []
    
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        # we have to traverse to know how many elements

        # O(n) to go through all
        # For each dfs we are doing O(log n) heappush work
        # Runtime == O(n logn)
        self.dfs(root)

        # O(n) worst case to get to the value
        num_seen = 0
        while num_seen < k:
            # O(log n) heappop
            curr = heapq.heappop(self.vals)
            num_seen += 1
        return(curr)

    def dfs(self, node: TreeNode):
        if not node:
            return

        self.num_elements += 1
        # O(logn) for heappush
        heapq.heappush(self.vals, node.val)
        self.dfs(node.left)
        self.dfs(node.right)