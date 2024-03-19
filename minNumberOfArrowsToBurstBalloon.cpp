#include <vector>;
using namespace std;
// 452
class Solution {
public:
    int findMinArrowShots(vector<vector<int>>& points) {
        int resp = 1;
        sort(points.begin(), points.end());
        int currMin = points[0][0];
        int currMax = points[0][1];
        for(vector<int> p : points) {
            int start = p[0];
            int end = p[1];
            cout << start << " " << end << endl;
            if(start > currMax or end < currMin) {
                resp++;
                currMin = p[0];
                currMax = p[1];
                // continue;
            }
            if(start <= currMax) {
                currMin = start;
                // continue;
            }
            if(end <= currMax) {
                currMax = end;
                // continue;
            }
        }
        return(resp);
    }
};