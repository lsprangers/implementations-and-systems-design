// 404
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int sumOfLeftLeaves(TreeNode* root) {
        int resp = dfs(root, false);
        return(resp);
    }
    int dfs(TreeNode* node, bool isLeft) {
        if(node == nullptr) {
            return(0);
        }
        int tmp_resp = 0;
        if(node->right != nullptr) {
            tmp_resp += dfs(node->right, false);
        }
        if(node->left != nullptr) {
            tmp_resp += dfs(node->left, true);
        }
        if(node->left == nullptr && node->right == nullptr && isLeft) {
            tmp_resp += node->val;
        } 
        return(tmp_resp);
    }
};