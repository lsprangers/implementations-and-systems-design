#include <vector>;
#include <queue>;
using namespace std;
//1642
class Solution {
public:
    int furthestBuilding(vector<int>& heights, int bricks, int ladders) {
        priority_queue<int, vector<int>, greater<int>> minHeap;

        // O(N)
        for(int i = 0; i < heights.size() - 1; i++) {
            int current_height = heights[i];
            int next_height = heights[i + 1];
            int difference = next_height - current_height;
            if(difference <= 0) {
                continue;
            }
            // Push is O(log L)
            minHeap.push(difference);
            if(minHeap.size() > ladders) {
                int smallest_use = minHeap.top();
                // Pop is O(log L)
                minHeap.pop();
                bricks -= smallest_use;
                if(bricks < 0) {
                    return(i);
                }
            }
        }
        // Time O(N log L)
        // Space O(L) b/c heap storage
        return(heights.size() - 1);
    }
};