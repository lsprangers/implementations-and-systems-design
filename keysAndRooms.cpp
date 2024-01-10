#include <vector>;
using namespace std;
// 841

class Solution {
public:
    unordered_map<int, vector<int>> umap;
    vector<int> seen;

    bool canVisitAllRooms(vector<vector<int>>& rooms) {
        int n_rooms = rooms.size();

        seen = vector(n_rooms, 0);
        for(int i = 0; i < n_rooms ; i ++) {
            umap[i] = rooms[i];
        }
        
        seen[0] = 1;
        dfs(0);
        int n_visited = std::reduce(seen.begin(), seen.end());

        return(n_visited == n_rooms);
    }
    void dfs(int node) {
        for(int neighbor : umap[node]) {
            if(seen[neighbor] == 0) {
                seen[neighbor] = 1;
                dfs(neighbor);
            }
        }
    }
};