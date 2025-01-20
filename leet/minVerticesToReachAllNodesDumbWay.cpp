// 1557
class Solution {
public:
    set<int> helper;
    set<int> seen;
    unordered_map<int, vector<int>> graph;

    vector<int> findSmallestSetOfVertices(int n, vector<vector<int>>& edges) {
        for(vector<int> edge : edges) {
            graph[edge[0]].push_back(edge[1]);
        }
        for(int key = 0; key < n; key++) {
            // skip if we've seen
            if(seen.find(key) != seen.end()) {
                continue;
            }
            dfs(key);
            helper.insert(key);
        }

        vector<int> resp(helper.begin(), helper.end());
        return(resp);

    }

    // doesn't matter if dfs or bfs
    void dfs(int node) {
        if(helper.find(node) != helper.end()) {
            helper.erase(node);
            return;
        }
        for(int neighbor : graph[node]) {
            if(seen.find(neighbor) == seen.end()) {
                seen.insert(neighbor);
                dfs(neighbor);
            }
        }
    }
};