#include <vector>;
using namespace std;
// 198
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        if(n < 3) {
            return(
                *max_element(begin(nums), end(nums))
            );
        }
        vector<int> tabb = vector(n+1, 0);
        tabb[0] = nums[0];
        tabb[1] = max(nums[0], nums[1]); //this got me
        for(int i = 2; i < n; i++) {
            tabb[i] = max(
                tabb[i - 1],
                nums[i] + tabb[i - 2]
            );
        }
        return(tabb[n - 1]);
    }
};