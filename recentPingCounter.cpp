#include <queue>;
using namespace std;

class RecentCounter {
public:
    queue<int> reqs;
    RecentCounter() {}
    
    int ping(int t) {
        reqs.push(t);

        int lowerBound = t - 3000;
        while(reqs.front() < lowerBound) {
            reqs.pop();
        }
        return(reqs.size());
    }
};