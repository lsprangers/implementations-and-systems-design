#include <queue>;
#include <vector>;
using namespace std;
//1338

class SortSecondary {
public:
    bool operator()(vector<int> below, vector<int> above)
    {
        // This puts vector with largest second column, first
        // So if the heap is at { {1, 2}, {1, 4}} and we 
        //  add in {2, 3} it is { {2, 3}, {1, 2}, {1, 4}}
        //  and at that point {2,3}[1] > {1,2}[1] so false
        //  is returned, and the 2 things are swapped
        if (below[1] < above[1]) {
            return true;
        }
 
        return false;
    }
};

class Solution {
public:
    int minSetSize(vector<int>& arr) {
        int n = arr.size();
        int n_half = floor((double)n / 2); //10 should be 5, 11 should be 5 
        int resp = 0;

        unordered_map<int, int> umap;
        for(int num : arr) {
            umap[num]++;
        }
        priority_queue<vector<int>, vector<vector<int>>, SortSecondary> maxHeap;
        for (const auto & [ key, value ] : umap) {
            maxHeap.push({key, value});
        }
        while(!maxHeap.empty() && n_half < n) {
            vector<int> thisSet = maxHeap.top();
            maxHeap.pop();
            n -= thisSet[1];
            resp++;
        }
        return(resp);

    }
};