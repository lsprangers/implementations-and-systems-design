#include <queue>;
using namespace std;
//973

class ComparePointsToOrigin {
public:
    vector<int> origin = {0, 0};
    bool operator()(vector<int> below, vector<int> above)
    {
        double below_diff = sqrt( ((below[0] - origin[0]) * (below[0] - origin[0])) + ((below[1] - origin[1]) * (below[1] - origin[1])));
        double above_diff = sqrt( ((above[0] - origin[0]) * (above[0] - origin[0])) + ((above[1] - origin[1]) * (above[1] - origin[1])));
        // This will put all points furthest from origin at top of the heap
        if (below_diff < above_diff) {
            return true;
        }
 
        return false;
    }
};

class Solution {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int k) {
        priority_queue<vector<int>, vector<vector<int>>, ComparePointsToOrigin> maxHeap;
        for(vector<int> point : points) {
            maxHeap.push(point);
            if(maxHeap.size() > k) {
                maxHeap.pop();
            }
        }
        vector<vector<int>> resp;
        while(!maxHeap.empty()) {
            resp.push_back(maxHeap.top());
            maxHeap.pop();
        }
        return(resp);
    }
};