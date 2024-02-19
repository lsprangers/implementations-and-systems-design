#include <vector>;
#include <queue>;
using namespace std;
//2149

class Solution {
public:
    vector<int> rearrangeArray(vector<int>& nums) {
        queue<int> above;
        queue<int> below;
        for(int num : nums) {
            if(num > 0) {
                above.push(num);
            }
            else {
                below.push(num);
            }
        }

        vector<int> resp;
        bool flip = true;
        while(!above.empty() || !below.empty()) {
            int toPush = 0;
            if(flip) {
                toPush = above.front();
                above.pop();
            }
            else {
                toPush = below.front();
                below.pop();
            }
            flip = !flip;
            resp.push_back(toPush);
        }
        return(resp);
    }
};