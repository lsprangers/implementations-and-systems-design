#include <vector>;
#include <set>;
#include <queue>;
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
    unordered_map<int, vector<int>> umap;
    queue<int> q;
    set<int> seen;
    int resp = -1;

    int amountOfTime(TreeNode* root, int start) {
        if(root == nullptr) {
            return(0);
        }
        createGraph(root);
        q.push(start);
        seen.insert(start);
        traverse(start);
        return(resp);
    }

    void createGraph(TreeNode* curr) {
        if(curr->left != nullptr) {
            umap[curr->val].push_back(curr->left->val);
            umap[curr->left->val].push_back(curr->val);
            createGraph(curr->left);
        }
        if(curr->right != nullptr) {
            umap[curr->val].push_back(curr->right->val);
            umap[curr->right->val].push_back(curr->val);
            createGraph(curr->right);
        }
    }

    void traverse(int curr) {
        while(!q.empty()) {
            // this breadth logic
            resp++;
            int q_size = q.size();

            // node level logic
            for(int i = 0; i < q_size; i++) {
                int curr = q.front();
                q.pop();
                for(int neighbor : umap[curr]) {
                    // if we can't find it
                    if(seen.find(neighbor) == seen.end()) {
                        seen.insert(neighbor);
                        q.push(neighbor);
                    }
                }
            }

        }

    }
};