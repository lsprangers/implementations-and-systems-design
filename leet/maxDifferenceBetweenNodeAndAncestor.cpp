#include <algorithm>;
using namespace std;
// 1026
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
    int maxDiff = INT_MIN;
    int maxAncestorDiff(TreeNode* root) {
        int maxHere = INT_MIN;
        int minHere = INT_MAX;
        return(dfs(root, maxHere, minHere));
    }
    int dfs(TreeNode* node, int maxHere, int minHere) {
        if(node == nullptr) {
            return(maxHere - minHere);
        }
        int leftResponse = dfs(node->left, max(maxHere, node->val), min(minHere, node->val));
        int rightResponse = dfs(node->right, max(maxHere, node->val), min(minHere, node->val));

        return( max(leftResponse, rightResponse) );
    }
};