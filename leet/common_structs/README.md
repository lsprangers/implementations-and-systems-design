# Common Data Structures
This is a place I write common data structures so I can use them in design systems or coding practice

### Representing a Tree in a List
- For future examples below, representing a tree in a list is done by using the following rules:
    - For any node at index `i`:
        - Left child is at index `(2 * i) + 1`
        - Right child is at index `(2 * i) + 2`
        - Parent is given by `(i - 1) // 2`
    - To access the first non-leaf node, you would look at `n // 2 - 1` (integer division). For example, with 8 items (indices 0 to 7), the first non-leaf node is at index `7 // 2 - 1 = 2`.
    - The first non-root node would be at index 1 or 2, depending on the context.

## Binary Search Tree
The [BST README](/dsa/trees%20&%20graphs/README.md) has more info
|Type        | Insert     | Search     | Delete     |
|------------|------------|------------|------------|
|Best Case   | O(1)       | O(1)       | O(1)       |
|Average Case| O(logn)    | O(logn)    | O(logn)    |
|Worst Case  | O(n)       | O(n)       | O(n)       |

- Binary Search Tree has the property that for any node, all of it's right children are greater, and all of it's left children are less than the current nodes value
- Therefore, for a balanced search tree, we'd cut off 1/2 of the nodes during each traversal, so in total there's only log n traversals done
    - Logn because we go from 
        - 16 --> 8 --> 4 --> 2 --> 1
        - 2^4 --> 2^3 --> 2^2 --> 2^1 --> 2^0
        - log(16) = 4, log(8) = 3, log(4) = 2, log(2) = 1
- If the tree is unbalanced, then it becomes a linked list and the insert, search and delete become worst case every time

## Max Heap
The [Max Heap README](/dsa/heap%20&%20priority%20queue/README.md) has more info
|Type        | Insert     | Search (Max only)     | Delete     |
|------------|------------|------------|------------|
|Best Case   | O(logn)       | O(1)       | O(logn)       |
|Average Case| O(logn)    | O(1)       | O(logn)    |
|Worst Case  | O(logn)       | O(1)       | O(logn)       |

- A Max Heap is a tree data structure which is a bit more specific than a Binary Search Tree, the Max Heap structure has the largest node as the root node, and any node's children are less than it's node
- Allows us to get the max item very quickly, and insert and delete operations involve potentially rotating some items but altogether are O(logn) for worst, average, and best cases since we still traverse a general tree structure