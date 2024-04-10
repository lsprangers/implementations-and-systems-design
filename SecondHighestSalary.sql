-- 176
-- Write your MySQL query statement below
with salary_ranker as (
    select 
        id, salary, 
        dense_rank() OVER(ORDER BY salary desc) as salary_rank
    from Employee
)

select max(salary) as SecondHighestSalary
from salary_ranker
where salary_rank = 2
limit 1