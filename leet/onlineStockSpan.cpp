#include <vector>;
#include <stack>;
using namespace std;
// 901

class StockSpanner {
public:
    stack<int> stk;
    vector<int> spans;
    vector<int> prices;
    int idx = 0;
    StockSpanner() {}
    
    int next(int price) {
        int count = 1;
        
        while(!stk.empty() && price >= prices[stk.top()]) {
            count += spans[stk.top()];
            stk.pop();
        }
        stk.push(idx);
        spans.push_back(count);
        prices.push_back(price);
        idx++;
        return(count);
    }
};