#include <algorithm>;
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};
//111
class Solution {
public:
    int minDepth(TreeNode* root) {
        if(root == nullptr) {
            return(0);
        }
        return(dfs(root));
    }
    int dfs(TreeNode* node) {
        if(node == nullptr) {
            return(INT_MAX);
        }
        if(node->left == nullptr and node->right == nullptr) {
            return(1);
        }
        int leftResponse = dfs(node->left);
        int rightResponse = dfs(node->right);
        return (1 + min(leftResponse, rightResponse));
    }
};