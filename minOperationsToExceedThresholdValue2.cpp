// 3066
class Solution {
    public:
        int minOperations(vector<int>& nums, int k) {
            // greedy
            // smallest first
            priority_queue<long, vector<long>, greater<long>> minHeap(nums.begin(), nums.end());
    
            bool larger = false;
            int resp = 0;
    
            while(!larger && minHeap.size() > 0) {
                long x = minHeap.top();
                minHeap.pop();
                if(x >= k) {
                    return(resp);
                }
                long y = minHeap.top();
                minHeap.pop();
    
                resp++;
                minHeap.push(min(x, y) * 2 + max(x, y));
            }
    
            return(-1);
    
    
        }
    };