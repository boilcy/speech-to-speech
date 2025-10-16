#!/bin/bash

BLUETOOTH_MAC="58:EA:1F:FF:1C:19"
SERVICE_USER="liucy"
USER_ID=$(id -u "$SERVICE_USER")

# When running as a systemd --user service, XDG_RUNTIME_DIR is usually handled.
# But for robustness or standalone testing, it's good to have.
export XDG_RUNTIME_DIR="/run/user/$USER_ID"
export PULSE_SERVER="unix:$XDG_RUNTIME_DIR/pulse/native"

echo "$(date): Attempting to connect to Bluetooth speaker $BLUETOOTH_MAC for user $SERVICE_USER..."

# try to connect for 15 minutes
MAX_ATTEMPTS_CONNECT=180
ATTEMPT=0
CONNECTED=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS_CONNECT ]; do
    ATTEMPT=$((ATTEMPT+1))
    echo "$(date): Attempt $ATTEMPT of $MAX_ATTEMPTS_CONNECT to connect..."

    # No sudo needed here, as the script runs as $SERVICE_USER
    bluetoothctl connect "$BLUETOOTH_MAC" &> /dev/null
    
    if bluetoothctl info "$BLUETOOTH_MAC" | grep -q "Connected: yes"; then
        CONNECTED=1
        echo "$(date): Successfully connected to $BLUETOOTH_MAC."
        break
    fi

    sleep 5
done

if [ $CONNECTED -eq 0 ]; then
    echo "$(date): Failed to connect to $BLUETOOTH_MAC after $MAX_ATTEMPTS_CONNECT attempts."
    exit 1
fi

echo "$(date): Setting default sink for user $SERVICE_USER..."

echo "$(date): Bluetooth auto-connect script finished."
exit 0
