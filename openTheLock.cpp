#include <queue>;
#include <unordered_set>;
#include <iostream>;
using namespace std;
// 752

class Solution {
public:
    int openLock(vector<string>& deadends, string target) {
        if (find(deadends.begin(), deadends.end(), "0000") != deadends.end()) {
            return -1;
        }
        
        queue<pair<string, int>> q;
        q.push(pair("0000", 0));
        unordered_set<string> seen(deadends.begin(), deadends.end());

        while(!q.empty()) {
            int q_size = q.size();
            auto [node, steps] = q.front();
            q.pop();
            if(node == target) {
                return(steps);
            }
            for (string neighbor: neighbors(node)) {
                // Not seen, and not deadend
                if(seen.find(neighbor) == seen.end()) {
                    seen.insert(neighbor);
                    q.push(pair(neighbor, steps + 1));
                }
            }
        }
        return(-1);
    }

    vector<string> neighbors(string node) {
        vector<string> ans;
        // Loop through all 4 parts of current step
        for (int i = 0; i < 4; i++) {
            int num = node[i];
            // Get up and down for each slot
            for (int change: {-1, 1}) {
                int x = (num - '0' + change + 10) % 10 + '0';
                string neighbor = node;
                neighbor[i] = x;
                ans.push_back(neighbor);
            }
        }
        
        return ans;
    }    
};