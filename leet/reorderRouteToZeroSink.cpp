#include <set>;
using namespace std;
//1466

class Solution {
public:
    unordered_map<int, vector<int>> umap;
    set<vector<int>> roads;
    vector<bool> seen;
    int resp;

    int minReorder(int n, vector<vector<int>>& connections) {
        // Create HashMap O(n) - we don't need to, but will be for first go
        seen = vector(n, false);
        resp = 0;

        for(vector<int> conxn : connections) {
            umap[conxn[0]].push_back(conxn[1]);
            // make it undirected so we can traverse starting at 0
            //  and we'll check if some item is in connections so we know 
            //  to include it in response
            umap[conxn[1]].push_back(conxn[0]);
            roads.insert({conxn[0], conxn[1]});
        }

        // Need to make 0 a sink for everything
        // Traverse out from 0, and change all arrows back to 0
        // Check every node once, and run dfs on it to make sure
        //  all nodes are covered back to 0
        dfs(0);
        return(resp);
    }

    void dfs(int node) {
        if(seen[node]) {
            return;
        }
        seen[node] = true;
        for(int neighbor : umap[node]) {
            if(seen[neighbor]) {
                if( roads.find({node, neighbor}) == roads.end() ) {
                    resp++;
                }
            }
            dfs(neighbor);
        }
    }
};