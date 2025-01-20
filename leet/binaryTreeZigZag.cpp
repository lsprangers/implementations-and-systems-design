
#include <vector>;
#include <queue>;
using namespace std;

// 103
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
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        queue<TreeNode*> que;
        vector<vector<int>> resp;
        int helper = 0;

        if(root == nullptr) {
            return(resp); // {{}}
        }
        que.push(root);

        while(!que.empty()) {
            int numNodesThisLevel = que.size();
            vector<int> thisResp;

            for(int i = 0; i < numNodesThisLevel; i++) {
                TreeNode* thisNode = que.front();
                que.pop();
                thisResp.push_back(thisNode->val);

                if(thisNode->right != nullptr) {
                    que.push(thisNode->right);
                }
                if(thisNode->left != nullptr) {
                    que.push(thisNode->left);
                }                              
            }
            if(helper % 2 == 0) {
                reverse(thisResp.begin(), thisResp.end());
            }
            resp.push_back(thisResp);
            helper++;
        }
        return(resp);
    }
};