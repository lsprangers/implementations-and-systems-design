#include <queue>;
using namespace std;
// 1609

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
    bool isEvenOddTree(TreeNode* root) {
        queue<TreeNode*> que;
        que.push(root);
        int level = -1;

        while(!que.empty()) {
            level++;
            int minValThisLevel = INT_MIN;
            int maxValThisLevel = INT_MAX;
            int qSize = que.size();

            for(int i = 0; i < qSize; i++) {
                TreeNode* thisNode = que.front();
                que.pop();
                if(level % 2 == 0) {
                    if(thisNode->val <= minValThisLevel || thisNode->val % 2 == 0) {
                        return(false);
                    }
                    minValThisLevel = thisNode->val;
                }
                else {
                    if(thisNode->val >= maxValThisLevel || thisNode->val % 2 == 1) {
                        return(false);
                    }
                    maxValThisLevel = thisNode->val;
                }
                if(thisNode->left != nullptr) {
                    que.push(thisNode->left);
                }
                if(thisNode->right != nullptr) {
                    que.push(thisNode->right);
                }
            }
        }
        return(true);
    }
};