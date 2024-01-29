#include <queue>;
using namespace std;
//2126
class Solution {
public:
    bool asteroidsDestroyed(int mass, vector<int>& asteroids) {
        long longMass = (long)mass;
        priority_queue<int, vector<int>, greater<int>> minHeap(asteroids.begin(), asteroids.end());

        while(!minHeap.empty()) {
            int thisAsteroid = minHeap.top();
            minHeap.pop();
            if(thisAsteroid <= longMass) {
                longMass += thisAsteroid;
            }
            else {
                return(false);
            }
        }
        return(true);
    }
};