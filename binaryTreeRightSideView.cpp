#include <vector>;
#include <queue>;
using namespace std;
//199
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
    vector<int> rightSideView(TreeNode* root) {
        if (root == nullptr) {
            return vector<int>{};
        }

        vector<int> resp;
        queue<TreeNode*> q;
        q.push(root);

        

        while (!q.empty()) { 
            int nodesAtThisLevel = q.size();
            resp.push_back(q.back()->val);

            for(int i = 0; i < nodesAtThisLevel; i++) {
                TreeNode* thisNode = q.front();
                q.pop();

                if(thisNode->left != nullptr) {
                    q.push(thisNode->left);
                }
                if(thisNode->right != nullptr) {
                    q.push(thisNode->right);
                }
            }
        }
        return(resp);
    }
};