# Security

# CloudTrail
Allow us to log SDK, CLI, or Console calls / actions from IAM Users or IAM Roles

## Event Types
- ***Management***: Operations performed on resources in AWS accounts
    - Logged by default
    - Anything that alters resources in the account
        - Attaching IAM policy, setting up logging, setting up data routing rules, etc...
    - ***Read Events*** don't modify resources
    - ***Write Events*** *may* modify resources
        - Much more important, more destructure
- ***Data***: Data events like S3 object level activity - Put, Delete, Get, etc...
    - Not logged by default
    - Can separate Read and Write Events like above too
    - S3, Lambda API Invoke, etc...
- ***Insights***: Helps us to find anomalies in our account
    - Paid for resource
    - Things like:
        - Inaccurate resource provisioning
        - Burst of IAM policies
    - Only categorizes / analyzes Write Events
    - These events also appear in Cloudtrail - usually a best practice to route these event types to Event Bridge or SNS to notify via email or something more than just logging

## Common Architectures
- ***S3 + Athena***: We can send them to AWS S3 or Cloudwatch Logs for further processing
    - Logs stored for 90 days in CloudTrail
    - Log to S3 for long term retention
        - Some use Athena to query over S3
- ***EventBridge***: Since all API's go through CloudTrail, we can filter and route specific ones to EventBridge
    - CloudTrail filter send to EventBridge
    - Event Bridge to
        - Lambda for processing
        - SNS for sending email
        - Another account for further processing
    - Common checks for SNS
        - User assuming role
        - User changing security group ingress rule
        - etc...

### Delivery to S3
- Can write CloudTrail directly to S3 using native integrations without anything in the middle
    - SSE-S3 or SS3-KMS for in route encryption
    - From there lifecycle policy can move these into glacier tier
- New files in S3 can then trigger S3 Events which send `Drop File` or `File Trigger` notification that can be picked up by Lambda / SNS / SQS
    - Cloud Trail ***could go straight to SNS / SQS / Lambda***, but it wouldn't be long term durable sitting in S3
    - Why S3?
        - Enable versioning
        - Enable MFA deletion
        - Enable S3 Lifecycle policy for archiving
        - S3 objcet lock to stop deletion
        - SSE KMS encryption
        - SHA-256 Hashing for integrity validation!
        - Blah blah blah, that's why we can use S3 and not straight to Lambda / SQS
- Can have a "Central Security Account" that has these central S3's, and then each LOB Member account can send CloudTrail logs there
    - Useful for central cyber / tech groups, and then 
        - Need cross-account S3 bucket policy
    - Can create cross-account IAM role in central account so Member accounts can assume this role to read

### Cloudwatch Logs to Notification System
- We can stream CloudTrail directly to Cloudwatch Logs
- From here we can use Metric Filters, which has structured window filtering languages to create new filters
    - If a metric threshold is passed, we can setoff a Cloudwatch Alarm and send the message to SNS
    - This essentially allows us to create a Notification and Alerting system on AWS
- From SNS we could also pipe alarms to Opsgenie, Pagerduty, etc...

### Organizational Trail
- Management account can create a CloudTrail Trail over all Member accounts!
- Any CloudTrail logs pipe back to Management account
    - Then these can go to S3 in Management account, and the write directories are prefixed by the Member Account ID

### Reactivity / SLA
- Cloudwatch, in worst case, takes 15 minutes to deliver an event
- EventBridge can be triggered for any API call
- Cloudwatch logs get events streamed to it from CloudTrail, and metric filter is a streaming tumbling window
- Events are delivered to S3 every 5 minutes, but with this we get durability, integrity, and many other features for ensuring data protection

# KMS
AWS Key Management Service (KMS) is a way to use keys, certificates, and ***encryption in general*** across services and resources

- Policies:
    - All keys in KMS are governed by ***Key Policies*** which define who can use the key (authentication) and for what purposes (authorization)
- Symmetric: Use same key to encrypt and unencrypt data
    - Can't download key
    - Must call KMS API to use, you can't get the key unencrypted
- Asymmetric: Public (encrypt) and private (decrypt)
    - Can download public key, and usually you'd do so to send out to other users who can't access AWS
        - Internal users who can call KMS would just call the API to encrypt data with a specific key!
    - Usually used by users / apps outside of AWS that can't call AWS KMS API to encrypt data, so they'd download public key and encrypt data to send back to AWS account and then we can decrypt
- Types:
    - ***Customer Managed Keys***: Keys you directly create and use via KMS API
        - Can do rotation, key policy, and CloudTrail auditing
    - ***AWS Managed Keys***: Used by AWS services like S3, EBS, Redshift, etc whenever we say we want things encrypted at rest or in transit
        - Managed by AWS fully
        - Keys rotated once a year
        - Key Policy is viewable, and audit trail in CloudTrail
    - ***AWS Owned Keys***: Created and managed by AWS
        - Used by some AWS services for encryption
        - Used in multiple AWS accounts, but not specifically our AWS accounts
            - Used by AWS internally
            - Maybe for serverless?
        - What?
- Key Source
    - AWS KMS hosted, where KMS takes care of everything
    - ***Exteral***: Where we create keys and use sources outside of KMS, and we import it into KMS
    - ***CloudHSM***: Run CloudHSM and cryptographic operations directly on clusters in your own VPC. When someone calls KMS, it uses your high availability clusters (across regions / AZ's) to do cryptographic calculations
- KMS is multi-region, but it just copies the keys over physically
    - One primary, multiple followers
        - Each key can be managed independently
    - Allows us to encrypt + decrypt over different regions

# SSM Parameter Store
- Secure storage for configurations and secrets
    - Security via IAM
    - Encryption via KMS
    - Notifications with EventBridge + SNS
    - IaC with CloudFormation
- Store parameters with a hierarchy
    - Most parameters are private to us
    - AWS has some parameters they store publicly for everyone, like latest IAM versions
    - Advanced parameters:
        - Can assign TTL expiration to parameters
        - `ExpirationNotification` and other notifications can go via EventBridge
        - Can assign multiple policies at a time

# Secrets Manager
- Store secrets! Like passwords, API Keys, etc...
- Can force rotation of secrets every X days
    - Automatic generation of secrets on rotation, it uses a lambda function with cron job to run new insert / update
    - Native support with almost everything - RDS, ECS, Fargate, EKS, etc...
        - ECS task can pull secrets at boot time, create environment variables, and then use it to access RDS securely

## Sharing 
- Sharing secrets across accounts:
    - Would encrypt the secret with a KMS Key
    - Create a resource policy on the KMS Key that specifies the secondary account can Decrypt, and you can even specify `kms:viaService` which means the secondary account can only use the KMS key if it's using it to decrypt in a secrets manager call
    - Then give the secret in secrets manager a resource policy to let the secondary account `secretsManager:GetSecretValue`

## Secrets Store vs SSM Parameter Store
- Secrets manager is more expensive
    - Automatic rotation
        - This means every 30 days Secrets Manager spins up a lambda, changes password on RDS, and updates value in Secrets Manager
    - KMS encryption is mandatory
    - Lambda provided for AWS Services like RDS and Redshift  
- SSM Parameter Store is cheaper
    - Simpler API
    - No secret rotation, but can use lambda triggered by EventBridge
        - Every 30 days EventBridge would need to wake up, change RDS password, and update value in SSM Param Store and ensure encryption and IAM done
    - KMS optional
    - Can pull Secrets Manager secret using SSM Parameter Store API
- Secrets Manager is basically just SSM Parameter store with extra integrations and best practices established

# RDS Security
- RDS has multiple layers of security, encryption, and auth
- KMS encryption at rest for underlying EBS volumes / snapshots
    - Basically means for the actual block storage for a live RDS, or for snapshot data, we can encrypt it at rest via KMS
- Transparent Data Encryption (TDE) for Oracle and SQL Server
    - Apparently important to remember that TDE can only be done for managed database providers (Oracle, Microsoft)
- IAM Authentication for MySQL, MariaDB, and Postgres (basically the ope source ones). Therefore, we can do user authentication for the open source databases
- SSL encryption to RDS is possible for all databases
- All ***Authorization*** still happens in RDS no matter what - Authentication has many avenues, but Authorization is all apart of Database users and groups
- Can copy unencrypted RDS snapshot to an encrypted one
- CloudTrail cannot be used to track RDS queries
    - You'll need a different observability solution for database query logging

# SSL/TLS and MITM
- Secure Socket Layer (SSL) is used to encrypt connections
    - Client to Server
    - One or Two Way
- Transport Layer Security (TLS) is a newer version of SSL
- Basically all certs are TLS, but a lot of people refer to it as SSL
- Public SSL/TSL certs are provider by Certificate Authorities (CA)
    - SSL certs have an expiration date (you set) 
- Types:
    - ***Asymmetric*** Encryption is expensive (lots of CPU time to compute it)
    - ***Symmetric*** Encryption is cheaper, and only uses one key
        - Symmetric encryption uses the same key to encrypt and decrypt
    - So:
        - Asymmetric handshake is used to exchange a per-client Symmetric random key, and then Client and Server use Symmetric key to talk in the future
        - This means that the asymmetric handshake is used strictly to create a shared session key, which can be used as a symmetric key to encrypt and decrypt information from the client to the server and vice versa
- The ***master key (or session key)*** in the SSL/TLS flow is a symmetric encryption key that is shared between the client and the server after the handshake process. It is used to encrypt and decrypt all subsequent communication between the client and server during the session.

![SSL Flow](./images/ssl.png)

## SSL/TLS Flow

- Everything from 1-3 is public plaintext, and everything afterwards, from 4+, is encrypted

1. **Client Hello**:
    - The client sends a "hello" message to the server, including:
        - Supported cipher suites (encryption algorithms)
        - A randomly generated value (**client random**)

2. **Server Hello**:
    - The server responds with:
        - Its own randomly generated value (**server random**)
        - Its SSL/TLS certificate, which contains the server's **public key**

3. **Certificate Verification**:
    - The client verifies the server's SSL/TLS certificate using a trusted Certificate Authority (CA) or other means
    - If the certificate is valid, the handshake continues

4. **Master Key Generation**:
    - The client generates a **pre-master key** (or directly a master key, depending on the protocol version)
    - The pre-master key is encrypted using the server's **public key** (from the SSL certificate) it sent in 2.2
    - The encrypted pre-master key is sent to the server 
        - No one snooping could decrypt what this pre-master key is at this point, only the server can

5. **Master Key Decryption**:
    - The server uses its **private key** to decrypt the pre-master key
    - Both the client and server then derive the **master key** from the pre-master key, the client random, and the server random
        - Server and Client Random could hypothetically be known by a MITM, but the pre-master key would be unreadable

6. **Session Encryption**:
    - The master key is used to derive symmetric encryption keys for encrypting and decrypting all subsequent communication
    - Both the client and server switch to symmetric encryption for the rest of the session

7. **Optional Client Authentication**:
    - If mutual authentication is required, the server may request the client's SSL/TLS certificate for verification

8. **Secure Communication**:
    - The client and server use the symmetric encryption keys (derived from the master key) to securely exchange data

---

### **Key Points About the Master Key**
- The master key is **never transmitted in plaintext**.
- It is derived from the **pre-master key**, **client random**, and **server random**.
- It ensures that all communication after the handshake is encrypted and secure.

---

### **Updated SSL Flow Diagram**
```plaintext
Client Hello (Client Random)  --->  Server Hello (Server Random + SSL Certificate)
       <---  Server Certificate Verification
Client generates Pre-Master Key
Client encrypts Pre-Master Key with Server Public Key
       --->  Encrypted Pre-Master Key sent to Server
Server decrypts Pre-Master Key with Private Key
Both derive Master Key (from Pre-Master Key + Randoms)
Secure communication begins using Symmetric Encryption
```

## SSL on Load Balancers
- SSL Server Name Indication (SSL SNI) solves the problem of hosting multiple SSL Certificates for multiple websites on one computer
    - One single VM can host websites 1-5, and each can have it's own SSL Certificate registered with the CA
    - Without SNI, servers wouldn't know which certificate to present during SSL handshake
- Then, when a client goes to connect to one of the 5 websites, it can specify that hostname and the web server will use the corresponding SSL Certificate in the asymmetric handshake
- This is how both ALB, NLB, and Cloudfront are able to do SSL certification and offloading for multiple backend services
    - ***SSL Offloading*** is the idea that ALB, NLB, or CloudFront can do SSL handshake, encryption, and decryption, and that logic can be removed from the application itself
    - Each LB / Cloudfront will host a certificate for each of the backend services, and will use that for SSL verification and encryption / decryption
    - The SSL Certificates are usually tied to target groups, where the target groups are related to hostname URL's
    - This ***does not work for Classic Load Balancers***, and for those we'd need to host a load balancer per domain, and have the singular SSL certificate on each load balancer

### Preventing MITM Attacks
- Don't use HTTP, use HTTPS
    - This will ensure traffic is encrypted, and any "server" you're talking to has been validated with a CA
    - Since data is encrypted with HTTPS, attacker can't read or modify data
- Use a DNS with DNSSEC compliance
    - Domain Name System Security Extensions (DNSSEC) adds more layers of security to DNS by ensuring the responses are authentic
    - There's a second attack vector, even with HTTPS, where a DNS entry is forged / fake and we still use SSL to try to reach `goodwebsite.com` and DNS ends up routing us to the incorrect bad one and it's still valid with the CA
        - This is known as ***DNS Spoofing***
    - Can bypass this by using a DNS Resolver thats DNSSEC compliant, and Amazon Route 53 is DNSSEC compliant, and even has KMS out of the box encryption for DNS queries

### SSL on Load Balancer Architecture
- Using ALB:
    - Setup ALB with SSL Cert(s) and for any incoming HTTPS request the ALB can do SSL handshake and then ALB sends HTTP data to auto-scaled target group of ECS tasks
- Maybe we want to use SSL directly onto an EC2 instance
    - TCP to NLB, and route requests to EC2 instance level
    - HTTPS from NLB to EC2
        - Can setup user scripts on EC2 so that at boot time it retrieves SSL Certs from SSM Parameter Store, where Certs are protected by IAM permissions
    - This is risky
        - Storing certs on EC2's, those EC2s exposed to internet, IAM roels for getting plaintext SSL certs, etc...
    - We can have the exact same architecture, except we replace SSM Parameter store with CloudHSM, and we use CloudHSM to do SSL offloading
        - SSL PK never leavs HSM device
        - CPU on EC2 doesn't get used to do decryption
        - Must setup a Cryptographic User (CU) on the CloudHSM device

# AWS Certificate Manager ACM
- AWS ACM can host your publicly created certifiacte, and it can help provision and renew public SSL certificates for you free of cost
- ACM loads SSL Certificates on the following integrations:
    - Load Balancers (including ones from ElasticBeanstalk)
    - CloudFront Distributions
    - APIs on API GW
- Uploading your own public certifiactes
    - Must verify public DNS
    - Must be issued by a trusted public CA
- Creating private certificates
    - For internal applications
    - You must create your own private CA
    - Your applications must trust your private CA
- Certificate Renewal
    - Automatically done if provisioned by ACM
    - Any manually upload cert must be renewed manually
- ACM is ***regional***
    - Therefore each region needs its own SSL Cert, and you can't copy SSL Certs across regions

# CloudHSM
- KMS gives us software
- HSM is when AWS provisions the hardware only
    - Hardware Security Module (HSM)
    - User manages the encryption keys entirely
    - HSM device is tamper resistant
    - Symmetric and Asymmetric encryption
    - Common pattern and good option for using SSE-C encryption on S3
- IAM permissions on HSM Cluster for CRUD operations
    - Everything else for authorization is on the CloudHSM cluster
    - It's similar to open source RDS securirty - IAM does authentication, cluster for authorization
- CloudHSM Cluster should be deployed for highly available, durable deployment across multiple AZ's
- Comparison
    - HSM is single tenant, KMS is multi-tenant
    - Both compliant
    - Both do symmetric and asymmetric 
        - HSM does Hashing and KMS doesn't
    - KMS does key replication across regions, HMS can do VPC peering across regional VPCs for cross region clusters / access
    - HSM is what allows TDE in Oracle and SQL Server