#include <vector>;
using namespace std;
// 513
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
    int maxDepth;
    TreeNode* leftMostNode;

    int findBottomLeftValue(TreeNode* root) {
        if(root == nullptr) {
            return(-1);
        }
        leftMostNode = root;
        maxDepth = 0;
        dfs(root, maxDepth);
        return(leftMostNode->val);
    }
    void dfs(TreeNode* node, int currDepth) {
        if(node == nullptr) {
            return;
        }
        if(currDepth > maxDepth) {
            maxDepth = currDepth;
            leftMostNode = node; // since we're doing left first, this should be fine
        }
        if(node->left != nullptr) {
            dfs(node->left, currDepth + 1);
        }
        if(node->right != nullptr) {
            dfs(node->right, currDepth + 1);
        }        
    }
};