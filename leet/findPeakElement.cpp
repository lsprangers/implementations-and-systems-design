// 162
#include <vector>;
using namespace std;

class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        int idx = 0;
        if(nums.size() < 2 || nums[0] > nums[1]) {
            return(0);
        }
        if(nums[nums.size() - 1] > nums[nums.size() - 2]) {
            return(nums.size() - 1);
        }
        while(true) {
            if(nums[idx] > nums[idx + 1] && nums[idx] > nums[idx - 1]) {
                return(idx);
            }
            idx++;
        }
    }
};