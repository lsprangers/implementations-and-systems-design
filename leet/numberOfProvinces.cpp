#include <vector>;
using namespace std;
//547

class Solution {
public:
    unordered_map<int, vector<int>> graph;
    vector<bool> seen;
    
    int findCircleNum(vector<vector<int>>& isConnected) {
        // createGraph - O(n^2) time complexity O(n) space
        int n = isConnected.size();
        seen = vector(n, false);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (isConnected[i][j] == 1) {
                    graph[i].push_back(j);
                    graph[j].push_back(i);
                }
            }
        }

        // This will loop over all unseen,
        //  and dfs will mark anything in
        //  the component as seen
        // So if everything is connected to
        //  0 then we'll have 1 province, 
        // We are guaranteed to cover all nodes
        // O(n) time
        int provinces = 0;
        for(int k = 0; k < n; k++) {
            if(!seen[k]) {
                seen[k] = true;
                provinces++;
                dfs(k);
            }
        }
        return(provinces);
    }

    void dfs(int pvince) {
        for(int neighbor : graph[pvince]) {
            if(!seen[neighbor]) {
                seen[neighbor] = true;
                dfs(neighbor);
            }
        }
    }
};