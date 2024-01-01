#include <queue>;
using namespace std;

class MovingAverage {
public:
    queue<double> q;
    double mAvg = 0.0;
    double mSum;
    int count = 0;
    int window;
    MovingAverage(int size) {
        window = size;
    }
    
    double next(int val) {        
        count++;
        q.push(val);
        mSum += val;
        if(count > window) {
            mSum -= q.front();
            q.pop();
            count = window;
        }
        return((double) mSum / count);
        
    }
};