#include <condition_variable>

using namespace std;

class Foo {
    std::mutex _mutex;
    int _butler = 'a';
    std::condition_variable _cv;
public:
    Foo() { }

    void first(function<void()> printFirst) {
        
        // printFirst() outputs "first". Do not change or remove this line.
        std::lock_guard<std::mutex> lock(_mutex);
        _butler = 'b'; 
        printFirst();        
        _cv.notify_all();  
    }

    void second(function<void()> printSecond) {
        
        // printSecond() outputs "second". Do not change or remove this line.
        std::unique_lock<std::mutex> lk(_mutex);
        _cv.wait(lk, [this] { return _butler == 'b'; });
        _butler = 'c';
                
        printSecond();        
        
        lk.unlock();
        _cv.notify_one();     
    }

    void third(function<void()> printThird) {
        
        // printThird() outputs "third". Do not change or remove this line.
        std::unique_lock<std::mutex> lk(_mutex);
        _cv.wait(lk, [this] { return _butler == 'c'; });
                
        printThird();        
        
        lk.unlock();    
    }
};