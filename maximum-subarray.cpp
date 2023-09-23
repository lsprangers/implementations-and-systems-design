class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int max_all = INT_MIN;
        int max_sub = 0;
        for(int n : nums) {
            max_sub += n;
            max_all = max(max_all, max_sub);
            // If the subarray is negative there's no use for it
            //      and so we reset it to 0
            max_sub = max(max_sub, 0);
        }
        return max_all;
    }
};