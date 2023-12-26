#include <vector>;
using namespace std;

class Solution {
public:
    int minStartValue(vector<int>& nums) {
        vector<int> cumSum;
        cumSum.push_back(nums[0]);
        int minn = nums[0];
        for(int i = 1; i < nums.size(); i++){
            cumSum.push_back(nums[i] + cumSum[i-1]);
            minn = min(minn, nums[i] + cumSum[i-1]);
        }
        return(max(1, 1-minn));
    }
};