#include <queue>;
using namespace std;
//1167

class Solution {
public:
    int connectSticks(vector<int>& sticks) {
        // O(n logn) build time
        // because O(n) loop, and O(log n) insert time
        priority_queue<int, vector<int>, greater<int>> minHeap(sticks.begin(), sticks.end());
        int cost = 0;

        // While loop is O(n) since each operation removes 1 element
        // pops are O(2)
        // insert is O(log n)
        while(minHeap.size() > 1) {
            int firstMin = minHeap.top();
            minHeap.pop();
            int secondMin = minHeap.top();
            minHeap.pop();
            cost += firstMin + secondMin;
            minHeap.push(firstMin + secondMin);
        }
        return(cost);

        // Time is O(n logn + n logn + 2)
    }
};