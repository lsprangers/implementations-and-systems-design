#include <set>;
#include <vector>;
using namespace std;
//2368
class Solution {
public:
    int resp = 0;
    set<int> restrictor;
    set<int> seen;
    unordered_map<int, vector<int>> umap;
    int reachableNodes(int n, vector<vector<int>>& edges, vector<int>& restricted) {
        if(edges.size() < 1) {
            return(0);
        }

        // for O(1) lookup, O(n) to create
        for(int restrictr : restricted) {
            restrictor.insert(restrictr);
        }

        
        // Create graph O(n) time and space
        for(vector<int> edge : edges) {
            umap[edge[0]].push_back(edge[1]);
            umap[edge[1]].push_back(edge[0]);
        }

        dfs(0);
        return(resp);

    }
    void dfs(int node) {
        if(restrictor.find(node) != restrictor.end() || seen.find(node) != seen.end()) {
            return;
        }
        else {
            seen.insert(node);
            resp++;
            for(int neighbor : umap[node]) {
                dfs(neighbor);
            }
        }
    }
};