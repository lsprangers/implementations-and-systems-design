#include <vector>;
#include <stack>;
using namespace std;
// 496
class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        vector<int> resp;
        unordered_map<int, int> nextGreater;
        stack<int> helper;

        // O(n) time, O(n) space for map
        for(int n2 : nums2) {
            if (helper.empty()) {
                helper.push(n2);
            }
            else {
                while(!helper.empty() && n2 > helper.top()) {
                    int toCreate = helper.top();
                    helper.pop();
                    nextGreater[toCreate] = n2;
                }
                helper.push(n2);
            }
        }
        // O(n) time, amortized O(1) space w/ section above
        while(!helper.empty()) {
            int lastOnes = helper.top();
            helper.pop();
            nextGreater[lastOnes] = -1;
        }

        // O(m) time where m < n
        for(int n1 : nums1) {
            resp.push_back(nextGreater[n1]);
        }



        return(resp);
    }
};