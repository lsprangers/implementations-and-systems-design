//2206
class Solution {
    public:
        bool divideArray(vector<int>& nums) {
            int n = nums.size() / 2;
            unordered_map<int, int> freqs;
            set<int> unique_nums(nums.begin(), nums.end());
    
            for(int num : nums) {
                if(freqs.find(num) != freqs.end() and freqs[num] != 0) {
                    freqs[num]--;
                }
                else {
                    freqs[num]++;
                }
            }
    
            for(auto unum : unique_nums){
                if(freqs[unum] != 0){
                    return(false);
                }
            }
            return(true);
        }
    };