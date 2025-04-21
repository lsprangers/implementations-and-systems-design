# Historic Local Auth
To get users to authenticate to your services, usually you'd just store a DB with usernames and passwords

Most of the time this just becomes downright annoying, and at the end of the day most people don't want to create Yet Another Username Password (YAUP)

We can move away from this model towards one where user authentication is done via dedicated, federated services...aka logging into one app with user/pass auth of another app

This other apps / provider is called an ***Identity Provider (IdP)***

Typical IdP providers include Google, Facebook, Twitter, and Apple

Key Components:
- Identity Provider (IdP)
    - Trusted service that handles user auth itself
    - Issues tokens like JSON Web Tokens (JWT) a.k.a ID Tokens, Access Tokens, etc...
- User:
    - User or app that's trying to authenticate
    - They login / authenticate to the IdP
- Desired Server:
    - Backend server hosting the protected resources (API, Database, Data Warehouse)
    - Verifies the token issued by the IdP to grant or deny access
        - IdP might send a random token...doesn't mean that user is even for you

## OpenID Connect
OpenID Connect is a way to ensure standard protocols and architecture around IdP provisioning

Most ID Tokens are JWT's

OpenID Connect Workflow:
- User initiation:
    - User initiates login to desired server / service
    - Desired service routes request to IdP login page
- User authenticates to IdP
    - User is verified by IdP
- IdP issues token
    - IdP will generate:
        - JWT Token which encodes and includes username, email, etc...
        - Access Token which grants access to resources on Desired Service
        - TODO: Both???
- User receives tokens
    - ID Token used to authenticate and identify the user
    - Access Token used to access authenticate resources
- User sends Access Token to Desired Service
- Desired Service grants access

[User] --> [Client] --> [IdP] --> [Client] --> [Resource Server]
   |            |           |         |             |
   |   Login    |  Auth +   | Tokens  | Access API  |
   |            |  Tokens   |         |             |s

Example:
- A user logs into a web app (client) using their Google account (IdP).
- Google authenticates the user and sends an ID Token and Access Token back to the web app.
- The web app uses the ID Token to display the user’s profile (e.g., name, email).
- The web app uses the Access Token to call a backend API (resource server) to fetch additional data.


