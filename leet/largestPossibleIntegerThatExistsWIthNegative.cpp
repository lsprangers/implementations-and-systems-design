class Solution {
public:
    int findMaxK(vector<int>& nums) {
        if(nums.size() < 2) {
            return(-1);
        }
        // sort descending
        // O(n log n)
        std::sort(nums.begin(), nums.end(), std::greater<>());
        int idx = 0;
        set<int> pos;
        // O(n)
        while(idx < nums.size() && nums[idx] >= 0) {
            pos.insert(nums[idx]);
            idx++;
        }


        // sort ascending
        // O(k logk) : k<=n
        std::vector<int> negatives(nums.begin() + idx, nums.end());      
        std::sort(negatives.begin(), negatives.end());
        idx = 0;

        // O(k)
        while(idx < negatives.size()) {
            if(pos.find(abs(negatives[idx])) != pos.end()) {
                return(abs(negatives[idx]));
            }
            idx++;
        }

        // O(n logn + klogk + k + n)
        return(-1);
    }
};