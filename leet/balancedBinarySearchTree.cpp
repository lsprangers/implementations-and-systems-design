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
    vector<TreeNode*> inorder_vec;
    TreeNode* balanceBST(TreeNode* root) {

        // O(n) to traverse inorder
        // inorder is left-root-right
        inorder(root);
        
        // O(n) to build from sorted array
        TreeNode* final = buildBalanced(0, inorder_vec.size() - 1);

        // O(2n) final
        return(final);
    }

    void inorder(TreeNode* node) {
        if(node == nullptr) {
            return;
        }

        // left-root-right
        inorder(node->left);
        inorder_vec.push_back(node);
        inorder(node->right);
    }

    TreeNode* buildBalanced(int start, int end) {
        // nullptr is acceptable pointer
        if(start > end) {
            return(nullptr);
        }

        // get the middle node, and recursively point to left
        //   and right nodes
        int mid = (start + end)/2;
        TreeNode* root = inorder_vec[mid];

        root->left = buildBalanced(start, mid - 1);
        root->right = buildBalanced(mid+1, end);

        return(root);
    }
};