# Table of Contents
- []

# Divide And Conquer
The divide and conquer family of DSA is all around splitting up the problem into subproblems that are easily solvable, and then merging at the end (if needed)

## GCD and LCM
Euclidean Algorithm: 

$\text{if } b \neq 0: GCD(A, B) = GCD(B, A \mod m)$

So we recursively break down our 2 components, A and B, until one of them is 0 which ultimately gives us the final result