// 624
class Solution {
public:
    int maxDistance(vector<vector<int>>& arrays) {
        // idx & number
        // O(2n) space
        priority_queue<pair<int, int>, vector<pair<int, int>>> maxHeap;
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minHeap;

        // O(n) iterations, with O(2 logn) inserts
        // So O(n logn) runtime
        for(int i = 0; i < arrays.size(); i++) {
            int array_size = arrays[i].size();
            maxHeap.push(pair(arrays[i][array_size - 1], i));
            minHeap.push(pair(arrays[i][0], i));
        }
        // if same index, find next
        pair<int, int> currMin = minHeap.top();
        pair<int, int> currMax = maxHeap.top();

        if(currMin.second == currMax.second) {
            int maxx = currMin.first;
            maxHeap.pop();
            int minn = currMax.first;
            minHeap.pop();
            pair<int, int> secondMin = minHeap.top();
            pair<int, int> secondMax = maxHeap.top();            
            return(
                max(
                    abs(secondMin.first - minn),
                    abs(maxx - secondMax.first)
                )
            );
        }
        return(
            abs(currMax.first - currMin.first)
        );

       
    }
};