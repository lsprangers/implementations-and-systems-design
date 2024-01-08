#include <queue>;
using namespace std;
// 1302

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
    int deepestLeavesSum(TreeNode* root) {
        queue<TreeNode*> que;
        int resp = 0;

        if(root == nullptr) {
            return(resp);
        }
        que.push(root);

        while(!que.empty()) {
            int numNodesThisLevel = que.size();

            // new response for this level
            resp = 0;

            for(int i = 0; i < numNodesThisLevel; i++) {
                TreeNode* thisNode = que.front();
                que.pop();
                resp += thisNode->val;

                if(thisNode->left != nullptr) {
                    que.push(thisNode->left);
                }
                if(thisNode->right != nullptr) {
                    que.push(thisNode->right);
                }                
            }
        }
        return(resp);

    }
};