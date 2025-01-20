
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};
// 112
class Solution {
public:
    int target;
    
    bool hasPathSum(TreeNode* root, int targetSum) {
        target = targetSum;
        return dfs(root, 0);
    }
    
    bool dfs(TreeNode* node, int curr) {
        if (node == nullptr) {
            return false;
        }
        if (node->left == nullptr and node->right == nullptr) {
            return(curr + node->val == target);
        }
        bool leftResult = dfs(node->left, curr + node->val);
        bool rightResult = dfs(node->right, curr + node->val);

        return(leftResult or rightResult);
    }
};