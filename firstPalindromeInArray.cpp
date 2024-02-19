#include <vector>;
#include <string>;
using namespace std;
//2108
class Solution {
public:
    string firstPalindrome(vector<string>& words) {
        // Loop is O(n * m) == O(n)
        // where n is the number of words and
        //  m is the max length of a word
        for(string word : words) {
            if(checkPalindrome(word)) {
                return(word);
            }
        }
        return("");
    }
    bool checkPalindrome(string word) {
        // Loop is O(n / 2) == O(n)
        // indexation is O(1)
        int left = 0;
        int right = word.size() - 1;
        while(left <= right) {
            if(word[left] != word[right]) {
                return(false);
            }
            left++;
            right--;
        }
        return(true);
    }
};