// 236

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    bool found_p = false;
    bool found_q = false;
    TreeNode* LCA = nullptr;
    
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        bool found = dfs(root, p, q);
        return(LCA);
    }
    bool dfs(TreeNode* node, TreeNode* p, TreeNode* q) {
        if(node == nullptr) {
            return(false);
        }

        int left = dfs(node->left, p, q) ? 1 : 0;
        int right = dfs(node->right, p, q) ? 1 : 0;

        int mid = (node == p || node == q) ? 1 : 0;

        if(mid + left + right >= 2) {
            LCA = node;
        }

        return(mid + right + left > 0);

    }
};