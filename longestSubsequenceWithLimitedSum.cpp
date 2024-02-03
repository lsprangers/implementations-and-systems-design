#include <vector>;
using namespace std;
//2389

class Solution {
public:
    vector<int> pfx_sum;
    vector<int> answerQueries(vector<int>& nums, vector<int>& queries) {
        sort(nums.begin(), nums.end());
        int s = nums.size();

        pfx_sum.push_back(nums[0]);
        for(int i = 1; i < nums.size(); i++) {
            pfx_sum.push_back(pfx_sum[i - 1] + nums[i]);
        }

        vector<int> resp;
        for(int query : queries) {
            resp.push_back(bsearch(query) + 1);
        }
        return(resp);
    }

    int bsearch(int query) {
        int left = 0;
        int right = pfx_sum.size() - 1;
        while(left <= right) {
            int mid = left + (right - left) / 2;
            if(pfx_sum[mid] == query) {
                return(mid);
            }
            else if(pfx_sum[mid] > query) {
                right = mid - 1;
            }
            else {
                left = mid + 1;
            }
        }
        return(right);
    }
};