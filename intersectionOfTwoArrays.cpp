#include <vector>;
#include <set>;
using namespace std;
// 349
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        set<int> one(nums1.begin(), nums1.end());
        set<int> two(nums2.begin(), nums2.end());
        set<int> intersect;
        set_intersection(one.begin(), one.end(), two.begin(), two.end(),
                 std::inserter(intersect, intersect.begin()));
        vector<int> resp;
        std::copy(intersect.begin(), intersect.end(), std::back_inserter(resp));
        return(resp);
    }
};