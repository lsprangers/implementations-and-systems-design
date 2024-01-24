#include <queue>;
using namespace std;
// 347

class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        // O(n) to create hashMap
        unordered_map<int, int> umap;
        for (int num: nums) {
            umap[num]++;
        }

        // O(n) loop, and in each loop we do at most O(log k) operations
        //  since the heap never goes above size K
        priority_queue<pair<int, int>> maxHeap;
        for (auto [key, val]: umap) {
            maxHeap.push(pair(-val, key)); //{-val, key}
            if(maxHeap.size() > k) {
                maxHeap.pop();
            }
        }

        // O(k) to iterate + O(1) pop
        vector<int> resp;
        while(maxHeap.size() > 0) {
            resp.push_back(maxHeap.top().second);
            maxHeap.pop();
        }
        return(resp);

        // Time is O(n + n log k + k)
        // Space is O(n + k)
    }
};