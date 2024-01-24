#include <queue>;
using namespace std;
//215

class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        priority_queue<int, vector<int>, greater<int>> maxHeap;
        //O(n * log k) time
        //  Each loop is n
        //  In each loop we do log k work by inserting
        for(int num : nums) {
            maxHeap.push(num);
            while(maxHeap.size() > k) {
                maxHeap.pop();
            }
        }
        return(maxHeap.top());
    }
};