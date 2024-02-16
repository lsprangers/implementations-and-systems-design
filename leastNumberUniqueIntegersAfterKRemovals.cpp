#include <vector>;
using namespace std;
//1481
class SortSecondary {
public:
    bool operator()(vector<int> below, vector<int> above)
    {
        // This puts vector with largest second column, first
        // So if the heap is at { {1, 2}, {1, 4}} and we 
        //  add in {2, 3} it is { {2, 3}, {1, 2}, {1, 4}}
        //  and at that point {2,3}[1] > {1,2}[1] so false
        //  is returned, and the 2 things are swapped
        if (below[1] > above[1]) {
            return true;
        }
 
        return false;
    }
};

class Solution {
public:
    int findLeastNumOfUniqueInts(vector<int>& arr, int k) {
        // Get count
        unordered_map<int, int> umap;
        for(int num : arr) {
            umap[num]++;
        }

        priority_queue<vector<int>, vector<vector<int>>, SortSecondary> minHeap;
        for(auto [num, count] : umap) {
            minHeap.push({num, count});
        }

        int thisNumCount = 0;
        while(k >= thisNumCount) {
            vector<int> currentMin = minHeap.top();
            // cout << "min" << currentMin[0] << " " << currentMin[1] << "\n";
            minHeap.pop();
            thisNumCount = currentMin[1];
            k -= thisNumCount;
            // cout << k << " " << thisNumCount << "\n";
        }

        int rest = k < 0 ? 1 : 0;
        return(rest + minHeap.size());


    }
};