// problem 100
#include <vector>
using namespace std;

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
    bool isSameTree(TreeNode* p, TreeNode* q) {
        // cover both nulls
        if(p == nullptr and q == nullptr) {
            return(true);
        }
        // if both aren't null, but one is, return false
        if(p == nullptr or q == nullptr) {
            return(false);
        }
        // check equivalence   
        if(p->val != q->val) {
            return(false);
        }
        // recurse
        return(
            isSameTree(p->left, q->left) 
            &&
            isSameTree(p->right, q->right) 
        );
    }
};