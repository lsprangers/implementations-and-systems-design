#include <algorithm>;
using namespace std;

//1448
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
    int numGood = 0;
    int goodNodes(TreeNode* root) {
        dfs(root, root->val);
        return(numGood);
    }
    
    int dfs(TreeNode* node, int maxVal) {
        if(node == nullptr) {
            return(0);
        }
        if(node->val >= maxVal) {
            numGood++;
        }
        int leftResponse = dfs(node->left, max(maxVal, node->val));
        int rightResponse = dfs(node->right, max(maxVal, node->val));
        return(1);
    }
};