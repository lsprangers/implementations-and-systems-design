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
    int findBottomLeftValue(TreeNode* root) {
        queue<TreeNode*> que;
        TreeNode* leftmost_this_level;
        // BFS will be slower than DFS

        que.push(root);
        while(!que.empty()) {
            int q_size = que.size();
            leftmost_this_level = que.front();
            for(int i = 0; i < q_size; i++) {
                TreeNode* thisNode = que.front();
                que.pop();
                if(thisNode->left != nullptr) {
                    que.push(thisNode->left);
                }
                if(thisNode->right != nullptr) {
                    que.push(thisNode->right);
                }
            }
        }
        return(leftmost_this_level->val);
    }
};