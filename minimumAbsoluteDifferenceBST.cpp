#include <vector>;
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};
//530
class Solution {
public:
    int resp = INT_MAX;
    vector<int> ara;
    int getMinimumDifference(TreeNode* root) {
        
        io_dfs(root);
        for(int i = 1; i < ara.size(); i++) {
            resp = min(resp, ara[i] - ara[i-1]);
        }
        return(resp);
    }

    void io_dfs(TreeNode* node) {
        if(node == nullptr) {
            return;
        }
        
        io_dfs(node->left);
        ara.push_back(node->val);
        io_dfs(node->right);
    }
};