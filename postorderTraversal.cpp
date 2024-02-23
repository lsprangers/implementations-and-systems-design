#include <vector>;
using namespace std;
//145
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
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> resp;
        dfs(root, resp);
        return(resp);
    }
    void dfs(TreeNode* node, vector<int>& resp) {
        if(node == nullptr) {
            return;
        }
        dfs(node->left, resp);
        dfs(node->right, resp);
        resp.push_back(node->val);
    }
};