# Cost Allocation
Basically just Tagging

## AWS Generated Cost Allocation Tags
- Automatically applied to resources you create
- Start with prefix `aws:` 
- Not applied to resourecs created before activation
- User Tags:
    - User defined tags
    - Start with prefix `user:`
- Cost Allocation Tags only show up in Billing Console

## Tag Editor
- Allows you to manage tags of multiple resources at once
- Can add / update / delete tags
- Search tagged / untagged resources in all AWS Regions

## Trusted Advisor
- No need to install anything
- Gives high level account assessment
- Analyzes and provides recommendation for:
    - Cost optimization
    - Service limits
    - Operational excellence
    - Performance
    - Security
    - Fault tolerance
- Core checks and recs to all customers
- Can enable weekly emails from UI
- Full trusted advisors
    - Ability to set CW alarms when reaching limits
    - Programmatic access to AWS Support API
- Support Plans:
    - Basic
        - 7 core checks
    - Developer
        - Paid for
        - 7 core checks
    - Business
        - Full set of checks
        - Programmatic access
    - Enterprise
        - Same as business, with extra other things
- Good to Know:
    - Can check if S3 is public, but can't check if S3 objects inside of bucket are public
        - Use EventBridge / S3 Events instead
    - Service limits
        - Limits can only be monitored in Trusted Advisor
        - Cases must be created manually in AWS Support Centre to increase limits
        - OR use AWS Service Quotas service
            - AWS Service Quotas helps us to get notifications around quotas near a certain threshold
            - Helps us to not get throttled
            - Can create CW Alarms over these events when we're about to approach them

## EC2 Launch Types & Savings
- ***On Demand Instances:*** Short workload, predictable pricing, reliable
- ***Spot Instances:*** Good for short cheap workloads, but AWS can reclaim capacity
- ***Reserved:***
    - Reserved Instances: long workloads
    - Convertible Reserved Instances: Long workloads with flexible instances
- ***Dedicated Instances:*** No other customer will share your hardware
- ***Dedicated Hosts:*** Book an entire physical server, control instance placement, and control many other aspects of host
    - Great for software licenses that oparte at core or socket level
    - Can define Host Affinity so that instance reboots are kept on same host

## AWS Savings Plan
- New pricing model to get a discount based on long-term usage
- Commit to a certain type of usage
    - $10 / hour for 1 to 3 years
    - Anything beyond is based on the on-demand pricing
- EC2 Instace Savings Plan
    - Up to 72% savings
    - Select instance family, lock to specific region, and lock in OS
- Compute Savings Plan
    - Up to 66% savings (smaller than EC2, but more flexiility)
    - Ability to move between instance family, region, compute type, OS, and tenancy
- Sagemaker Savings Plan
    - Up to 64% savings on sagemaker workloads

## S3 Storage Classes
- All S3 is 11 9's durability
- S3 Standard General Purpose
    - 99.99% Availability
    - Frequently accessed data
    - Low latency and high throughput
- S3 Standard Infrequent Access (IA)
    - Charged for extra usage, but static storage is cheaper
    - Less accessed, but requires rapid access
    - 99.9% availability
    - Use for DR and backups
- S3 One Zone-Infrequent Access
    - Cheaper infrequent access, but if we lose a zone then we can't get them back
    - High durability inside of one AZ
    - 99.5% availability
- S3 Intelligent Tiering
    - Pay for monitoring, but takes care of movement between storage types
    - Small monthly monitoring fee and auto-tiering fee
    - Moves objects between Access Tiers based on usage
    - Tiers:
        - *Frequent access:* default
        - *IA:* not accessed for 30 days
        - *Archive IA:* not accessed for 90 days
        - *Archive Access:* not accessed for 90 - 700+ days (configurable)
        - *Deep Archive:* not accessed for 180 - 700+ days (configurable)
- Glacier
    - All are based on how fast we can get them vs how expensive they are
        - Low cost object storage
    - Instant retrieval
        - Millisecond retrieval
        - Minimum storage duration of 90 days
    - Flexible retrieval
        - Expedited: 1-5 minutes
        - Standard: 3-5 hours
        - Bulk: 5-12 hours
        - Minimum storage duration of 90 days
    - Deep archive
        - Standard: 12 hours
        - Bulk: 48 hours
        - Minimum storage duration of 180 days        
- S3 Life Cycle Configurations / Rules allow us to transition between tiers
- Compressing objects saves space
- S3 requester pays
    - Forces person requesting object to pay, not you
    - You pay for storage cost, but they pay for egress out
    - Need to ensure S3 bucket policy so that user is authenticated via IAM
        - Can't use assume role
        - If it's an assumed role, then it's the owner of the account role that pays
        - So if you create the role, and they assume it, then you'll end up paying

## AWS Budgets
- Create budgets and send alarms when cost exceeds the budget
- 4 types of budgets: 
    - Usage
    - Cost
    - Reservation
    - Savings Plan
- For Reserved Instances
    - Track utilization
    - Supports EC2 elasticache, RDS, and Redshift
- Up to 5 SNS notifications per budget
- Can filter by many tag and metadata values
- Same options as AWS Cost Explorer
- Runs actions for you when budget exceeds a certain threshold
    - Budget service itself will run some compute
    - Supports 3 action types:
        - Apply IAM policy to a user, group, or IAM role
        - Apply an SCP to an OU
        - Stop EC2 or RDS instances

## AWS Compute Optimizer
- Can help us with right sizing EC2 instances based on metrics around CPU, RAM, Network, Disk, etc...
- Setup CW Agent on EC2 instances and send metrics to AWS Compute Optimizer