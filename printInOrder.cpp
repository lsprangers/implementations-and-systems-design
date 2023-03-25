// 1114 - concurrency

using namespace std;

class FooBar {
private:
    int n;
    std::mutex mtx;
    std::condition_variable cv;
    bool is_foo = true;

public:
    FooBar(int n) {
        this->n = n;
    }

    void foo(function<void()> printFoo) {
        // unique locks can be locked and unlocked
        //  whereas lock guards are destroyed after use
        //...Both are better than pure locks which are
        //      exposed to exception issues
        for (int i = 0; i < n; i++) {
            std::unique_lock<std::mutex> ulk_foo(mtx);
            cv.wait(ulk_foo,[&]()
            {
                return is_foo;
            });
        	// printFoo() outputs "foo". Do not change or remove this line.
        	printFoo();
            is_foo = false;
            ulk_foo.unlock();
            cv.notify_one();
        }
    }

    void bar(function<void()> printBar) {
        for (int i = 0; i < n; i++) {
            std::unique_lock<std::mutex> ulk_bar(mtx);
            cv.wait(ulk_bar,[&]()
            {
                return !is_foo;
            });
        	// printBar() outputs "bar". Do not change or remove this line.
        	printBar();
            is_foo = true;
            ulk_bar.unlock();
            cv.notify_one();
        }
    }
};