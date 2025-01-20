// 129
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
    vector<vector<int>> nums;
    int resp;
    int sumNumbers(TreeNode* root) {
        vector<int> start = {};
        dfs(root, start);
        for(vector<int> num : nums) {
            int nSize = num.size();
            int tmpResp = 0;
            for(int i = 0; i < nSize; i++) {
                cout << num[i];
                tmpResp += num[i] * pow(10, nSize - 1 - i);
            }
            resp+=tmpResp;
        }
        return(resp);
    }
    void dfs(TreeNode* node, vector<int> someNums) {
        if(node == nullptr) {
            return;
        }


        someNums.push_back(node->val);
        if(node->left == nullptr && node->right == nullptr) {
            nums.push_back(someNums);
            return;
        }
        dfs(node->left, someNums);
        dfs(node->right, someNums);
        
    }
};