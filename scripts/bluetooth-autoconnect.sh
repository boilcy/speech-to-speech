#!/bin/bash

# 蓝牙音箱的MAC地址
BLUETOOTH_MAC="58:EA:1F:FF:1C:19"
USER="$(whoami)"

echo "Attempting to connect to Bluetooth speaker $BLUETOOTH_MAC..." | systemd-cat -t bluetooth-autoconnect

# 循环尝试连接，直到成功或达到最大尝试次数
MAX_ATTEMPTS=10
ATTEMPT=0
CONNECTED=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT+1))
    echo "Attempt $ATTEMPT of $MAX_ATTEMPTS..." | systemd-cat -t bluetooth-autoconnect

    # 通过 bluetoothctl 连接设备
    sudo -u "$USER" bluetoothctl connect "$BLUETOOTH_MAC" &> /dev/null
    
    # 检查是否连接成功
    if sudo -u "$USER" bluetoothctl info "$BLUETOOTH_MAC" | grep -q "Connected: yes"; then
        CONNECTED=1
        echo "Successfully connected to $BLUETOOTH_MAC." | systemd-cat -t bluetooth-autoconnect
        break
    fi

    sleep 5 # 等待5秒后重试
done

if [ $CONNECTED -eq 0 ]; then
    echo "Failed to connect to $BLUETOOTH_MAC after $MAX_ATTEMPTS attempts." | systemd-cat -t bluetooth-autoconnect
    exit 1
fi

echo "Setting default sink for user $USER..." | systemd-cat -t bluetooth-autoconnect

# 等待 PulseAudio/PipeWire 服务启动并检测到蓝牙设备
# 在Systemd服务中，pulseaudio通常需要一些时间来启动并识别新连接的蓝牙设备。
# 这里多加几次尝试，确保蓝牙设备作为音频接收器被识别。

SINK_SET=0
for i in {1..10}; do
    # 获取当前的默认sink，并检查是否为蓝牙sink
    CURRENT_DEFAULT_SINK=$(sudo -u "$USER" pactl info | grep "Default Sink" | awk '{print $3}')
    
    if [[ "$CURRENT_DEFAULT_SINK" == *"bluez_sink"* || "$CURRENT_DEFAULT_SINK" == *"$BLUETOOTH_MAC"* ]]; then
        echo "Bluetooth sink already set as default: $CURRENT_DEFAULT_SINK" | systemd-cat -t bluetooth-autoconnect
        SINK_SET=1
        break
    fi

    # 尝试设置蓝牙音箱为默认输出
    # 我们寻找名字包含蓝牙MAC地址的sink
    BLUETOOTH_SINK_NAME=$(sudo -u "$USER" pactl list sinks | grep -B 20 "$BLUETOOTH_MAC" | grep "Name:" | head -n 1 | awk '{print $2}')
    if [ -n "$BLUETOOTH_SINK_NAME" ]; then
        echo "Found Bluetooth sink: $BLUETOOTH_SINK_NAME, attempting to set as default..." | systemd-cat -t bluetooth-autoconnect
        sudo -u "$USER" pactl set-default-sink "$BLUETOOTH_SINK_NAME"
        if [ $? -eq 0 ]; then
            echo "Successfully set $BLUETOOTH_SINK_NAME as default sink." | systemd-cat -t bluetooth-autoconnect
            SINK_SET=1
            break
        else
            echo "Failed to set $BLUETOOTH_SINK_NAME as default sink." | systemd-cat -t bluetooth-autoconnect
        fi
    else
        echo "Bluetooth sink not yet available in PulseAudio/PipeWire. Retrying..." | systemd-cat -t bluetooth-autoconnect
    fi
    
    sleep 3 # 等待3秒后重试
done

if [ $SINK_SET -eq 0 ]; then
    echo "Failed to set Bluetooth speaker as default audio sink after multiple attempts." | systemd-cat -t bluetooth-autoconnect
    exit 1
fi

echo "Bluetooth auto-connect script finished." | systemd-cat -t bluetooth-autoconnect
exit 0
