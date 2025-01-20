#include <vector>;
#include <set>;
using namespace std;

class Solution {
public:
    unordered_map<int, vector<int>> umap;
    int resp = 0;
    set<int> seen;
    int countComponents(int n, vector<vector<int>>& edges) {
        // O(e) time where e = # edges
        for(vector<int> edge : edges) {
            umap[edge[0]].push_back(edge[1]);
            umap[edge[1]].push_back(edge[0]);
        }
        // O(v) - amortized since we'll visit each node only once
        for(int i = 0; i < n; i++) {
            if(seen.find(i) == seen.end()) {
                resp++;
                seen.insert(i);
                dfs(i);
            }
        }
        return(resp);
    }
    void dfs(int node) {
        // O(v) - amortized
        for(int next : umap[node]) {
            if(seen.find(next) == seen.end()) {
                seen.insert(next);
                dfs(next);
            }
        }
    }
};