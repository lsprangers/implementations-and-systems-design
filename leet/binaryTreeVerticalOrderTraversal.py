# 314
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def __init__(self):
        # root = 0, -1 +1 based on left/right
        # TODO: this can be further optimized to directly insert into resp
        self.levels = defaultdict(list)
        self.min_col = 0
        self.max_col = 0
        self.resp = []

    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        self.dfs(root, 0, 0)
        
        for col_lvl in range(self.min_col, self.max_col + 1):
            self.levels[col_lvl].sort(key=lambda x:x[0])
            self.resp.append([val for row, val in self.levels[col_lvl]])
        
        # jfc
        return(self.resp if self.resp != [[]] else [])

        
    def dfs(self, node, row_level, col_level):
        if node == None:
            return
        
        # same row and col - order left to right 
        #   means preorder or inorder
        self.levels[col_level].append((row_level, node.val))
        self.min_col = min(self.min_col, col_level)
        self.max_col = max(self.max_col, col_level)

        self.dfs(node.left, row_level + 1, col_level - 1)
        self.dfs(node.right, row_level + 1, col_level + 1)
        return