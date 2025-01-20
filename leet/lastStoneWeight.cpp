#include <vector>;
#include <queue>;
using namespace std;
// 1046
class Solution {
public:
    int lastStoneWeight(vector<int>& stones) {
        // O(n) build time
        if(stones.size() < 1) {
            return(-1);
        }
        priority_queue<int> heap(stones.begin(), stones.end());
        while(heap.size() > 1) {
            int top = heap.top();
            heap.pop();
            int second = heap.top();
            heap.pop();
            int smash = top - second;
            if(smash != 0) {
                heap.push(smash);
            }
        }
        if(heap.size() > 0) {
            return(heap.top());
        }
        return(0);
    }
};