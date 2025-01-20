// 1750
class Solution {
public:
    int minimumLength(string s) {
        int holder = 0;
        int left = 0;
        int right = s.size() - 1;

        // O(n) time complexity - string access O(1), and we loop through the string once
        // O(1) space complexity
        while(left < right && s[left] == s[right]) {
            while(s[left] == s[holder] && left <= right) {                
                left++;
            }
            while(s[right] == s[holder] && left <= right) {                
                right--;
            }
            holder = left;
        }
        return(
            right - left + 1
        );
    }
};