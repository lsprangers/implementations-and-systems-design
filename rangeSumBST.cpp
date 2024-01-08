
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};
// 938
class Solution {
public:
    int resp = 0;
    int rangeSumBST(TreeNode* root, int low, int high) {
        io_dfs(root, low, high);
        return(resp);
    }
    void io_dfs(TreeNode* node, int low, int high) {
        if(node == nullptr) {
            return;
        }

        io_dfs(node->left, low, high);
        if(node->val >= low and node->val <= high) {
            resp += node->val;
        }
        io_dfs(node->right, low, high);

        
    }
};