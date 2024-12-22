// 55
#include <vector>;
using namespace std;

class Solution {
public:
    bool canJump(vector<int>& nums) {
        vector<bool> canMake(nums.size(), false);
        canMake[nums.size() - 1] = true;
        int lastTrue = nums.size() - 1;
        for(int right = nums.size() - 1; right > -1; right--) {
            if(right + nums[right] >= lastTrue) {
                lastTrue = right;
                canMake[right] = true;
            }
        }
        return(canMake[0]);
    }
};