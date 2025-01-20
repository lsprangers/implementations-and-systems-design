// 791
class Solution {
public:
    string customSortString(string order, string s) {
        unordered_map<char, int> freq;
        string resp;

        // O(m)
        for(char c : s) {
            freq[c]++;
        }
        // O(n)
        queue<char> order_q;
        for(char o : order) {
            order_q.push(o);
        }
        // O(n)
        while(!order_q.empty()) {
            char curr = order_q.front();
            order_q.pop();
            int curr_count = freq[curr]; //0 if not exists
            // O(1)
            while(curr_count > 0) {
                resp += curr;
                curr_count--;
            }
            freq.erase(curr);
        }
        // get any other values that aren't in order
        // O(m)
        for(auto const& [key, val] : freq) {
            int curr_count = val;
            while(curr_count > 0) {
                resp += key;
                curr_count--;
            }     
        }
        return(resp);
    }
};