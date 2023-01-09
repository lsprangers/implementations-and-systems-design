# problem 145
from typing import Optional, List

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        if root:
            res.extend([root.val])
            left = self.preorderTraversal(root.left)
            right = self.preorderTraversal(root.right)
            res.extend(left)
            res.extend(right)
        
        return res