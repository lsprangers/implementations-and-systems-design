# High Level Solutions Architecture
What usually happens in AWS?

- ***DNS***: User needs to find services, so it uses Route53 DNS Layer
- ***Web Layer***:
    - ***Static***: User might go to CDN Cloudfront for static content
    - ***Dyanmic***: User might need to go to xLB for dynamic content
        - Dynamic content is served by compute, EC2, ECS, Fargate, Lambda, etc...
            - xLB can also source from CDN Cloudfront
        - This dynamic data typically needs data, that data comes from:
            - ***Database Layer*** for stateful content in tabular, document, or graph format
            - ***Caching / Session Layer*** for quick response of Database Layer
            - ***Static Data*** data might sit in EBS, EFS, or Instance Store (EC2)
            - ***Static Assets*** might sit in S3
        - These services might need to talk to each other in a ***Service Mesh*** which is communicated via SQS, SNS, Kinesis, MQ, or Step Functions
- All of these things have ***Authentication and Authorization*** via IAM 

# Compute

## EC2 Instance Types
- R: Appliactions that need lots of (R)AM - In Memory Caches
- C: Applications that need lots of (C)PU - Databases / Batch Compute
- M: Applications that are balanced with (M)edium / Balanced resources - General Web Apps
- I: Applications that need good local (I)/O - Databases
- G: Applications that need a (G)PU - ML + Video
- T2/T3: Burstable Instances (to capacity)
- T2/T3 Unlimited: Unlimited Burst

### EC2 Placement Groups
Can use ***Placement Groups*** so that certain configs are covered

- Group Strategies:
    - ***Cluster***: Clusters instances into low latency group in single AZ
        - Same rack, same AZ
        - 10GBpS bandwidth between instances
            - Means if rack fails, all nodes fail
        - Used for HPC, Big Data Hadoop, etc...
    - ***Spread***: Spread instances across underlying hardware (used for critical applications)
        - Different hardware across AZ's
        - Use multi-AZ, no simultaneous failure
        - EC2 on different physical failure
        - But limited to 7 instances per AZ per group
        - Used to maximize high availability critical apps
    - ***Partition***: Spreads instances across many different partitions (partitions rely on different sets of racks) within an AZ
        - Keep multiple partitions in same AZ, and then instances in a partition are still in different racks
        - Partition failures can affect multiple EC2's, but not all of them
        - EC2 instances get access to partition information as metadata
- Moving instances
    - Stop the instance
    - Use CLI command `modify-instance-placement`
    - Restart instance

### EC2 Instance Launch Types
- ***On Demand Instances***: Short workloads, predictable pricing, reliable, typical
- ***Spot Instances***: Short workloads that are much cheaper than On-Deamand
    - However, you can lose instances if the pool of them is low and an On-Demand request comes in
    - Useful for resilient applications, like Spark or Distributed app, that can handle if an instance goes down
- ***Reserved***: Minimum 1 year instance you fully reserve to be yours
    - Useful for long workloads
    - ***Convertible Instances*** allow you to change the instance type over time
    - Payment plans of full upfront, partial upfront, or none upfront give highest to lowest discounts
- ***Dedicated Instances***: Means no other customer can share hardware
    - May share underlying server itself with others
- ***Dedicated Hosts***: Means only your instance(s) are ever on the actual server
    - Useful when you need access to core level, kernel level, or full socket level applications
        - Typically for software licenses that operate at network I/O socket and file level
    - Can also define *host affinity* so that instance reboots all sit on the same underlying host (server)

#### Graviton
AWS Graviton Processosrs deliver the best price performance, and they are only on linux based instances

### EC2 Metrics
- CPU: Utilization + Credit Usage
- Network: In and Out
- Status Check: Instance and Systems status
    - Can have CloudWatch monitor our EC2 instances, and if there's an alarm for `StatusChecFailed_System`, we can use ***EC2 Instance Recovery***
    - EC2 Instance Recovery allows us to keep same Private, Public, and Elastic IP addresses, along with metadata and placement group
- Disk: Read/Write for Ops/Bytes
- RAM: ***RAM IS NOT INCLUDED IN AWS EC2 METRICS*** and must be sent from EC2 into CloudWatch metrics by the user

## HPC
High Performance COmputing is being pushed by AWS because the costs of doing it yourself are so large, and really groups want to use a ginormous number of clusters at once, and then run something, and then be done

- Data Mgmt and Transfer:
    - ***AWS Direct Connect***: We can move GBpS of data to the cloud over a private secure network
    - ***Snowball***: Moves PB of data to the cloud
    - ***AWS DataSync***: Move large amount of data between on-prem and S3, EFS, EBS, or FSx for Windows
- Compute and Networking:
    - EC2! We use CPU or GPU instances, with Spot or Spot Fleets for cheap, giant clusters
    - We use EC2 placement group of type `Cluster` to keep all of these instances on the same rack with 10 GBpS of networking out of the box
    - Networking:
        - ***ENI***: Elastic Network Interfaces are the typical networking interface on EC2
        - EC2 Enhanced Networking (SR-IOV):
            - Higher bandwidth, higher packets per second (PPS), and lower latency
            - ***ENA***: Allows us up to 100 GBps
            - Intel 82599 is legacy...10 GBpS
        - ***EFA***: The Elastic Fabric Adapter is an even more specific type of elastic network adapter
            - Improved ENA for HPC
            - Only works for Linux
            - Great for tightly coupled, inter-node workloads
            - Uses ***Message Passing Interface (MPI)*** Standards typical for HPC workloads
                - Also helps us to write SIMD calcs on GPU
            - Bypasses the underlying Linux OS to provide low latency transport across nodes
- Storage:
    - Instance attached storage
        - EBS: Scale up to 256k IOPS
        - Instance Store: Scales to millions of IOPS since it's linked to EC2
    - Network Storage:
        - S3: Large blob...not a filesystem
        - EFS: Scale IOPS based on total size
        - AWS FSx for Lustre
            - HPC optimized for millions of IOPS
            - SOmehow backed by S3
- Automation and Orchestration
    - ***AWS Batch*** supports multi-node parallel jobs, which enables you to run single jobs that span multiple EC2 instances
        - Easily schedule jobs and launch EC2 accordingly
    - ***AWS ParallelCluster*** 
        - Open source cluster manager tool to deploy HPC workloads on AWS
        - Config with text files
        - Automate creation of VPC, subnet, cluster type, and instances
        - Useful for researchers who don't wanna IaC