// 26
#include <set>;
using namespace std;
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        set<int> seen;
        int k = 0;
        int idx = 0;
        for(int i = 0; i < nums.size(); i++) {
            // haven't found
            if(seen.find(nums[i]) == seen.end()) {
                k++;
                nums[idx] = nums[i];
                idx++;
            }
            seen.insert(nums[i]);
        }
        return(k);
    }
};