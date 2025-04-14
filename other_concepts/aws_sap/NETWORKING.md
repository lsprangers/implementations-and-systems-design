# Networking
Networking is mostly around connecting networks that are in cloud and on prem, and then configuring those networks for our services!

Virtual Private Clouds (VPCs) are the cloud component for networking, and they define a set of addresses and subnets

Things to write later:
- Public subnet is just a subet with an internet gateway
- Network ACL
    - Access Control Lists talk about what kind of network can come in
    - They are stateless, subnet-level firewalls
    - Specify IP ranges, CIDR's, protocol, and port level infromation to deny and allow traffic
- Security Groups
    - Stateful instnace-level firewalsl that control inbound and outbound traffic on EC2 instances
    - They allow you to allow or deny traffic into instances themselves
    - Stateful means if inbound on a port is accepted, then outbound is automatiaclly allowed
