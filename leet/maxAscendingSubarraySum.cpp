// 1800
class Solution {
public:
    int maxAscendingSum(vector<int>& nums) {
        int resp = nums[0];
        int temp = nums[0];

        int left = 0;
        for(int right = 1; right < nums.size(); right++) {
            if(nums[right] <= nums[right - 1]) {
                left = right;
                temp = nums[right];
            }
            else {
                temp += nums[right];
            }
            resp = max(resp, temp);
        }

        return(resp);
    }
};