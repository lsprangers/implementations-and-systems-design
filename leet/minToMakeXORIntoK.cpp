// Using built-in I didn't know existed
class Solution {
public:
    int minOperations(vector<int>& nums, int k) {

        int numsXOR = 0;
        for(int num : nums) {
            numsXOR = (numsXOR ^ num);
        }

        int resp = 0;
        while(k || numsXOR) {
            // %2 gets last bit
            // XOR is 1 only if they're diff
            resp += ((k%2) ^ (numsXOR%2));

            // Remove last bit
            k /= 2;
            numsXOR /= 2;
        }

        return(resp);
    }
};



// Old solution using strings TLE
class Solution {
public:
    int minOperations(vector<int>& nums, int k) {
        vector<int> binaryk(22, 0);
        int idx = 0;
        while(k > 0) {
            int mod = k % 2;
            k /= 2;
            binaryk[idx] = mod;
            idx++;
        }
        // std::reverse(std::begin(binaryk), std::end(binaryk));

        vector<vector<int>> binary_reps(nums.size(), vector<int>(22, 0));
        for(int i = 0; i < nums.size(); i++) {
            int tmp = nums[i];
            int idx = 0;
            while(tmp > 0) {
                int mod = tmp % 2;
                tmp /= 2;
                binary_reps[i][idx] = mod;
                idx++;
            }
            // std::reverse(std::begin(binary_reps[i]), std::end(binary_reps[i]));
        }

        int resp = 0;
        for(int i = 0; i < 22; i++) {
            int want = binaryk[i];
            int curr = want;
            for(vector<int> others : binary_reps) {
                curr = (curr ^ others[i]);
            }
            resp += curr;
        }

        return(resp);
    }
};