#include <queue>;
using namespace std;
//1962

class Solution {
public:
    int minStoneSum(vector<int>& piles, int k) {
        priority_queue<int> maxHeap(piles.begin(), piles.end());
        for(int i = 0; i < k; i++) {
            int maxStone = maxHeap.top();
            maxHeap.pop();
            maxHeap.push(ceil((double)maxStone / 2));
        }
        int stoneSum = 0;
        while(!maxHeap.empty()) {
            stoneSum += maxHeap.top();
            maxHeap.pop();
        }
        return(stoneSum);
    }
};