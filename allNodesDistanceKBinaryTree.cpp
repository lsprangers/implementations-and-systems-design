#include <vector>;
#include <set>;
#include <queue>;
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

//863 
class Solution {
public:
    unordered_map<int, vector<TreeNode*>> umap;
    set<TreeNode*> seen;
    int kDist;
    vector<int> resp;

    vector<int> distanceK(TreeNode* root, TreeNode* target, int k) {

        // Need to do graph + bfs 
        // O(n) time and space
        buildGraph(root);

        queue<TreeNode*> que;
        que.push(target);
        bfs(que, k);
        return(resp);

    }
    void bfs(queue<TreeNode*> que, int k) {
        int level = -1;
        while(!que.empty()) {
            level++;
            int q_size = que.size();
            if(level > k) {
                return;
            }
            for(int i = 0; i < q_size; i++) {
                TreeNode* curr = que.front();
                seen.insert(curr);
                que.pop();
                if(level == k) {
                    resp.push_back(curr->val);
                }
                for(TreeNode* neighbor : umap[curr->val]) {
                    if(seen.find(neighbor) == seen.end()) {
                        que.push(neighbor);
                    }
                    
                }
            }
            
            
        }
    }
    void buildGraph(TreeNode* node) {
        if(node->left != nullptr) {
            umap[node->val].push_back(node->left);
            umap[node->left->val].push_back(node);
            buildGraph(node->left);   
        }
        if(node->right != nullptr) {
            umap[node->val].push_back(node->right);
            umap[node->right->val].push_back(node);
            buildGraph(node->right); 
        }  
    }
};