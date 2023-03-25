// 1116 - concurrency

class ZeroEvenOdd {
private:
    int n;
    std::mutex mtx;
    std::condition_variable cv;
    bool is_even = false;
    bool is_zero = true;

public:
    ZeroEvenOdd(int n) {
        this->n = n;
    }

    // printNumber(x) outputs "x", where x is an integer.
    void zero(function<void(int)> printNumber) {
        int i = 0;
        while(i < n) {
            std::unique_lock<std::mutex> ulk_zero(mtx);
            cv.wait(ulk_zero,[this]
            {
                return is_zero;
            });
            // printf("%i zero \n", 0);
            printNumber(0);
            i+=1;
            // is_even = !is_even;
            is_zero = false;
            ulk_zero.unlock();
            cv.notify_all();
        }
    }

    void even(function<void(int)> printNumber) {
        int j = 2;
        while(j <= n) {
            std::unique_lock<std::mutex> ulk_even(mtx);
            cv.wait(ulk_even,[this]
            {
                return !is_zero && is_even;
            });
            // printf("%i even  \n", j);
            printNumber(j);
            j+=2; 
            is_even = !is_even;
            is_zero = true;
            ulk_even.unlock();
            cv.notify_all();
        }     
    }

    void odd(function<void(int)> printNumber) {
        int k = 1;
        while(k <= n) {
            std::unique_lock<std::mutex> ulk_odd(mtx);
            cv.wait(ulk_odd,[this]
            {
                return !is_zero && !is_even;
            });
            // printf("%i odd  \n", k);
            printNumber(k);
            k+=2;
            is_even = !is_even;
            is_zero = true;
            ulk_odd.unlock();
            cv.notify_all();
        }               
    }
};