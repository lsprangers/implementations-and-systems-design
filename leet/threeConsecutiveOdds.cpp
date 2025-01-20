// 1550

// O(n) time O(1) space
class Solution {
public:
    bool threeConsecutiveOdds(vector<int>& arr) {
        int left = 0;
        int count = 0;
        while(left < arr.size()) {
            if(arr[left] % 2 != 0) {
                count++;
            }
            else {
                count = 0;
            }
            left++;

            
            if(count == 3) {
                return(true);
            }
        }
        return(false);
    }
};