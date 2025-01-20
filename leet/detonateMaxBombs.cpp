#include <vector>;
#include <set>;
#include <queue>;
using namespace std;
// 2101

class Solution {
public:
    unordered_map<int, vector<int>> umap;
    int maximumDetonation(vector<vector<int>>& bombs) {        

        // Build graph O(n^2)
        for(int i = 0; i < bombs.size(); i++) {
            for(int j = 0; j < bombs.size(); j++) {
                if(inside_range(bombs[j], bombs[i] ) ) {
                    umap[i].push_back(j);
                    // umap[j].push_back(i);
                }
            }
        }

        int resp = 0;
        // BFS O(n^2) because we don't restrict already visited between nodes
        for(int bomb_idx = 0; bomb_idx < bombs.size(); bomb_idx++){
            resp = max(resp, bfs(bomb_idx));
        }
        return(resp);

    }

    int bfs(int start_idx) {
        queue<int> q;
        set<int> seen;
        seen.insert(start_idx);
        q.push(start_idx);

        while(!q.empty()) {
            int this_bomb_idx = q.front();
            q.pop();
            for(int neighbor : umap[this_bomb_idx]) {
                if(seen.find(neighbor) == seen.end()) {
                    seen.insert(neighbor);
                    q.push(neighbor);
                }
            }           
        }
        return(seen.size());
    }

    bool inside_range(vector<int> neighbor, vector<int> center) {
        long x_diff = (long)center[0] - neighbor[0];
        long y_diff = (long)center[1] - neighbor[1];
        long dist = (long)center[2] * center[2];
        return(
            dist >= (x_diff * x_diff) + (y_diff * y_diff) 
        );
    }
};