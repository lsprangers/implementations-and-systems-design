#include <set>;
#include <queue>;
using namespace std;
//1306

class Solution {
public:
    bool canReach(vector<int>& arr, int start) {
        queue<int> idx_que;
        set<int> seen_idxs;
        int steps = -1;

        // Indexing is O(1)
        idx_que.push(start);
        while(!idx_que.empty()) {
            // Want to run for each breadth to keep track of steps
            int q_size = idx_que.size();
            steps++;
            for(int i = 0; i < q_size; i++) {
                int idx = idx_que.front();
                idx_que.pop();

                if(arr[idx] == 0) {
                    return(true);
                }

                if(seen_idxs.find(idx) == seen_idxs.end()) {
                    seen_idxs.insert(idx);
                    for(int next_idx : {idx + arr[idx], idx - arr[idx]}) {
                        if(-1 < next_idx && next_idx < arr.size()) {
                            idx_que.push(next_idx);
                        }
                    }
                }
            }
        }
        return(false);
    }
};