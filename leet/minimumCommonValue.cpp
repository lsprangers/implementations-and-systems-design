#include <vector>;
using namespace std;
// 2540
class Solution {
public:
    int getCommon(vector<int>& nums1, vector<int>& nums2) {
        int oneIdx = 0;
        int twoIdx = 0;
        while(oneIdx < nums1.size() && twoIdx < nums2.size()) {
            if(nums1[oneIdx] == nums2[twoIdx]) {
                return(nums1[oneIdx]);
            }
            else if(nums1[oneIdx] < nums2[twoIdx]) {
                oneIdx++;
            }
            else {
                twoIdx++;
            }
        }
        return(-1);
    }
};