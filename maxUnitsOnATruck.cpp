#include <queue>;
#include <vector>;
using namespace std;
//1710
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
    int maximumUnits(vector<vector<int>>& boxTypes, int truckSize) {
        int resp = 0;
        priority_queue<vector<int>, vector<vector<int>>, SortSecondary> maxHeap(boxTypes.begin(), boxTypes.end());
        while(!maxHeap.empty() && truckSize > 0) {
            vector<int> thisBoxSet = maxHeap.top();
            maxHeap.pop();
            int numBoxes = thisBoxSet[0];
            int boxUnits = thisBoxSet[1];
            while(numBoxes > 0 && truckSize >= 1) {
                numBoxes--;
                truckSize -= 1;
                resp += boxUnits;
            }
        }
        return(resp);
    }
};