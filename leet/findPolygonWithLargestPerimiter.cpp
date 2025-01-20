#include <vector>;
using namespace std;
//2971

class Solution {
public:
    long long largestPerimeter(vector<int>& nums) {
        vector<long long> cSum;
        // O(N logN) worst case
        sort(nums.begin(), nums.end());
        cSum.push_back(nums[0]);
        for(int i = 1; i < nums.size(); i++) {
            cSum.push_back(cSum[i - 1] + nums[i]);
        }

        long long maxSeen = -1;
        for(int j = 0; j < nums.size() - 1; j++) {
            if(cSum[j] > nums[j + 1]) {
                maxSeen = max(maxSeen, cSum[j + 1]);
            }
        }
        return(maxSeen);

    }
};