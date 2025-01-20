#include <vector>;
using namespace std;

// O(1) memory space and in-place change
// O(n) time complexity
class Solution {
public:
    void reverseString(vector<char>& s) {
        int left = 0;
        int right = s.size() - 1;
        while(left < right) {
            char leftChar = s[left];
            char rightChar = s[right];
            s[left] = rightChar;
            s[right] = leftChar;
            left++;
            right--;
        }
    }
};