#include <vector>;
using namespace std;

class Solution {
public:
    vector<int> getAverages(vector<int>& nums, int k) {
        vector<int> resp;
        if(nums.size() < ((k*2) + 1) ){
            for(int z = 0; z < nums.size(); z++){
                resp.push_back(-1);
            }
            return(resp);
        }
        
        vector<unsigned long long int> cumSum;
        cumSum.push_back(nums[0]);
        
        // get cumSum
        // O(n) time and O(n) space
        for(int i = 1; i < nums.size(); i++) {
            cumSum.push_back(nums[i] + cumSum[i-1]);
        }
        // push back initial -1
        for(int l = 0; l < k; l++) {
            resp.push_back(-1);
        }
        //main avg driver
        // O(n) time and space
        for(int j = k; j < nums.size() - k; j++) {
            int avg_here = ( (cumSum[j + k] - cumSum[j-k] + nums[j-k]) / ((k*2) + 1) );
            resp.push_back( avg_here );
        }
        // push back final -1
        for(int m = 0; m < k; m++) {
            resp.push_back(-1);
        }
        // Altogether O(2n) time and space
        return(resp);
    }
};