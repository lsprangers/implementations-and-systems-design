#include <algorithm>;
using namespace std;
//543
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    int resp = 0;
    int diameterOfBinaryTree(TreeNode* root) {
        if(root == nullptr) {
            return(0);
        }
        dfs(root);
        return(resp);
    }
    int dfs(TreeNode*  node) {
        if(node == nullptr) {
            return(0);
        }
        int leftResponse = dfs(node->left);
        int rightResponse = dfs(node->right);
        resp = max(resp, leftResponse + rightResponse);
        return(1 + max(leftResponse, rightResponse));
    }
};