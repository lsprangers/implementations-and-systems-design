// 3075
class Solution {
public:
    long long maximumHappinessSum(vector<int>& happiness, int k) {
        int sadness = 0;
        long long totalHappiness = 0;
        
        // Filling with Floyd's is O(n)
        // Filling one by one is O(N log N)
        priority_queue<int, vector<int>> maxHeap(happiness.begin(), happiness.end());
        while(k > 0) {
            totalHappiness += max(maxHeap.top() - sadness, 0);
            maxHeap.pop();
            sadness++;
            k--;
        }
        return(totalHappiness);
    }
};