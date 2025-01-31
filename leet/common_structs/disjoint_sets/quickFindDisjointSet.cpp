// In quickFind we store the root node itself instead of the parent node
//  therefore, the find() operation is O(1) 
//  but the union() operation is O(n)

//      0000
//     /    \
//    1      2
//   / \    / \
//  3   4  5   6

// Example:
// parent   = [0, 0, 0, 1, 1, 2, 2]
// root     = [0, 0, 0, 0, 0, 0, 0]
// So we see here that root just has 0's

// If we have a tree like this:
//    0
//    |    
//    1      2
//   / \    / \
//  3   4  5   6

// Example:
// parent   = [0, 0, 2, 1, 1, 2, 2]
// root     = [0, 0, 2, 0, 0, 2, 2]

// Then we see our find() is still O(1), but we'd need an 
// extra step to do union()

// Example:
// parent   = [0, 0, 0, 1, 1, 2, 2]
// root     = [0, 0, 2, 0, 0, 2, 2]
// After this, we need to update 5 and 6 to have
//  root[5] = 0 and root[6] = 0

#include <vector>;
#include <iostream>;
using namespace std;

class quickFind {
public:
    quickFind(int n) {
        // Constructor
        parent = vector<int>(n);
        root = vector<int>(n, 0);
        count = n;
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
        }
    }

    int find(int x) {
        // parent   = [0, 1, 2, 3, 4, 5]
        // rank     = [0, 0, 0, 0, 0, 0]
        return root[x];
    }

    void unionSet(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        // O(n) just iterate through and change
        if (rootX != rootY) {
            for (int i = 0; i < root.size(); i++) {
                if (root[i] == rootY) {
                    root[i] = rootX;
                }
            }
        }
    }

    bool connected(int x, int y) {
        return find(x) == find(y);
    }

private:
    vector<int> parent;
    vector<int> root;
    int count;
};


// Test Case
int main() {
    // for displaying booleans as literal strings, instead of 0 and 1
    cout << boolalpha;
    quickFind uf(10);
    // 1-2-5-6-7 3-8-9 4
    uf.unionSet(1, 2);
    uf.unionSet(2, 5);
    uf.unionSet(5, 6);
    uf.unionSet(6, 7);
    uf.unionSet(3, 8);
    uf.unionSet(8, 9);
    cout << uf.connected(1, 5) << endl;  // true
    cout << uf.connected(5, 7) << endl;  // true
    cout << uf.connected(4, 9) << endl;  // false
    // 1-2-5-6-7 3-8-9-4
    uf.unionSet(9, 4);
    cout << uf.connected(4, 9) << endl;  // true

    return 0;
}