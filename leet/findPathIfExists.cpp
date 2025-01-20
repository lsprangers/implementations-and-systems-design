#include <vector>;
#include <set>;
using namespace std;
//1971

class Solution {
public:
    bool resp = false;
    unordered_map<int, vector<int>> umap;
    set<int> seen;
    int our_destination;
    bool validPath(int n, vector<vector<int>>& edges, int source, int destination) {
        our_destination = destination;
        if(edges.size() == 0) {
            return(true);
        }
        for(vector<int> edge: edges) {
            umap[edge[0]].push_back(edge[1]);
            umap[edge[1]].push_back(edge[0]);
        }
        dfs(source);
        return(resp);
    }
    void dfs(int node) {
        if(seen.find(node) == seen.end()) {
            seen.insert(node);
            for(int next: umap[node]) {
                if(next == our_destination) {
                    resp = true;
                    return;
                }
                dfs(next);
            }
        }
        
    }
};