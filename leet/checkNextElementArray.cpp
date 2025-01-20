#include <vector>;
using namespace std;

class Solution {
public:
    int countElements(vector<int>& arr) {
        // set<int> dis_nums(arr.begin(), arr.end());
        int countt = 0;
        for (int d_num : arr) {
            if(find(begin(arr), end(arr), d_num+1) != end(arr)) {
                countt++;
            }
        }
        return(countt);
    }
};