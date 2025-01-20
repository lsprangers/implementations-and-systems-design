#include <vector>;
using namespace std;
//300
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> tabb = vector(n, 1);
        for(int i = 1; i < n; i++) {
            int maxx = 0;
            for(int j = 0; j < i; j++) {
                if(nums[j] < nums[i]) {
                    maxx = max(maxx, tabb[j]);
                }
            }
            tabb[i] += maxx;
        }
        return(*max_element(begin(tabb), end(tabb)));
    }
};