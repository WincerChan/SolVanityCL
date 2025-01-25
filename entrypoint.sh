#!/bin/bash
set -e


# Setup SSH authorized_keys if avain_ssh is provided
if [ -n "$avain_ssh" ]; then
    mkdir -p /root/.ssh
    echo "$avain_ssh" > /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
fi
# Start the SSH daemon
/usr/sbin/sshd




# Sleep for 10 minutes to keep the container running after postcli
sleep 600
# Following the postcli command, we run the command passed to the docker run
# This will typically be a long-running command to keep the container alive
exec "$@"
