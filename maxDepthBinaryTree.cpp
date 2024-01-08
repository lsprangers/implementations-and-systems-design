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


class Solution {
public:
    int maxDepth(TreeNode* root) {
        int resp = 0;

        if(root == nullptr) {
            return(resp);
        }
        resp++;

        int leftDepth = maxDepth(root->left);
        int rightDepth = maxDepth(root->right);
        
        resp += max(leftDepth, rightDepth);
        return(resp);
    }
};