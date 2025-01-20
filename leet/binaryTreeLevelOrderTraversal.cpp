#include <vector>;
#include <queue>;
using namespace std;
//102
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> resp;
        if(root == nullptr) {
            return(resp);
        }
        
        queue<TreeNode*> que;
        que.push(root);

        while(!que.empty()) {
            int numNodesThisLevel = que.size();
            vector<int> thisResp;

            for(int i = 0; i < numNodesThisLevel; i++) {
                TreeNode* thisNode = que.front();
                que.pop();
                thisResp.push_back(thisNode->val);
                if(thisNode->left != nullptr) {
                    que.push(thisNode->left);
                }                      
                if(thisNode->right != nullptr) {
                    que.push(thisNode->right);
                }
                         
            }
            resp.push_back(thisResp);
        }
        return(resp);
    }
};