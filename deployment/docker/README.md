# Docker Deployment with SSH Tunnel

This directory contains Docker configurations for running the ClariGen services. It uses an SSH tunnel to securely access model servers hosted on a remote machine (`grace.cas.mcmaster.ca`).

## Prerequisites

- **Docker** and **Docker Compose** installed on your local machine.
- **SSH Private Key (`id_rsa`)**: You must have an SSH private key that allows you to log in to the remote server.

## SSH Tunnel Setup

To establish the secure connection, follow these steps carefully. Distinguish between actions performed on your **Host Machine** (your computer) and the **Remote Server**.

### 1. On Your Host Machine (Local)

The Docker container needs a copy of your private key to open the tunnel.

1.  **Prepare the Key File**:
    Copy your private key to this directory (`deployment/docker`) and name it exactly `id_rsa`.
    ```bash
    # Run in deployment/docker/
    cp ~/.ssh/id_rsa ./id_rsa
    ```
    *Note: If your key has a different name, copy it and rename it to `id_rsa`.*

2.  **Secure the Key**:
    Ensure the key is not readable by others.
    ```bash
    chmod 600 ./id_rsa
    ```

3.  **Generate Public Key** (If you don't have it):
    You will need the public key to authorize access on the server.
    ```bash
    ssh-keygen -y -f ./id_rsa > ./id_rsa.pub
    ```
    *Copy the content of `id_rsa.pub` for the next section.*

### 2. On The Remote Server

You need to authorize your key on the server `grace.cas.mcmaster.ca`.

1.  **Log in to the Remote Server**:
    Use your existing credentials or ask an administrator for access.
    ```bash
    ssh username@grace.cas.mcmaster.ca
    ```

2.  **Add to Authorized Keys**:
    Append the content of your **local** `id_rsa.pub` to the `~/.ssh/authorized_keys` file on the remote server.
    ```bash
    # On grace.cas.mcmaster.ca
    # Replace <YOUR_PUBLIC_KEY_CONTENT> with the actual text from id_rsa.pub
    echo "<YOUR_PUBLIC_KEY_CONTENT>" >> ~/.ssh/authorized_keys
    ```

3.  **Verify Permissions**:
    Ensure the `.ssh` directory and `authorized_keys` file have correct permissions on the server.
    ```bash
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys
    ```

### 3. Verify Connection (Critical)

Before starting Docker, test that your key works from your local machine:

```bash
# Run on your Host Machine
ssh -i ./id_rsa -o StrictHostKeyChecking=no username@grace.cas.mcmaster.ca echo "Success"
```
If this prints `Success`, you are ready to proceed.

## Running the Services

1.  **Start Docker Compose**:
    ```bash
    docker-compose up -d
    ```

2.  **Check Tunnel Status**:
    ```bash
    docker-compose logs -f tunnel
    ```
    If the container is running and not restarting strictly, the tunnel is active.

## Troubleshooting

### Error: "/root/.ssh/id_rsa is a directory"
This happens if you started Docker *before* creating the local `id_rsa` file. Docker creates a directory in its place.
**Fix:**
```bash
docker-compose down
rm -rf ./id_rsa      # Remove the directory
cp ~/.ssh/id_rsa .   # Copy the file again
docker-compose up -d
```

### Connection Refused / Permission Denied
- Ensure you copied the **Private Key** (starts with `-----BEGIN OPENSSH PRIVATE KEY-----`), not the public key.
- Ensure the public key is correctly added to `~/.ssh/authorized_keys` on `grace.cas.mcmaster.ca`.
