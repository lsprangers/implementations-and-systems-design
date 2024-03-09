#include <queue>;
using namespace std;
//215

class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        priority_queue<int, vector<int>, greater<int>> minHeap;
        //O(n * log k) time
        //  Each loop is n
        //  In each loop we do log k work by inserting
        for(int num : nums) {
            minHeap.push(num);
            while(minHeap.size() > k) {
                minHeap.pop();
            }
        }
        return(minHeap.top());
    }
};