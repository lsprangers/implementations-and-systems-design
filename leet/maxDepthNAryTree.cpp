/*
// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> children;

    Node() {}

    Node(int _val) {
        val = _val;
    }

    Node(int _val, vector<Node*> _children) {
        val = _val;
        children = _children;
    }
};
*/

class Solution {
public:
    int maxDepth(Node* root) {
        if(root==nullptr) {
            return(0);
        }
        return(dfs(root) + 1);
    }
    int dfs(Node* node) {
        if(node->children.size() < 1) {
            return(0);
        }
        int maxHere = 0;
        for(Node* child : node->children) {
            maxHere = max(
                dfs(child),
                maxHere
            );
        }
        return(maxHere + 1);
    }
};