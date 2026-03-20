#!/usr/bin/env bash
# Install a udev rule for the PCsensor FootSwitch so it is accessible without sudo.
# Run once: sudo bash scripts/install_footpedal_udev.sh
# Then unplug and replug the footpedal.

set -euo pipefail

RULE_FILE=/etc/udev/rules.d/99-footpedal.rules
NAME_HINT="PCsensor FootSwitch Keyboard"

# Find the matching event device via /sys (no device-open permission needed)
SYS_DEV=""
for dir in /sys/class/input/event*/; do
    name_file="$dir/device/name"
    if [[ -f "$name_file" ]] && grep -qi "$NAME_HINT" "$name_file"; then
        SYS_DEV="$dir/device"
        break
    fi
done

if [[ -z "$SYS_DEV" ]]; then
    echo "Error: no device matching '$NAME_HINT' found."
    echo "Check 'python scripts/test_footpedal.py --list' for available devices."
    exit 1
fi

VENDOR=$(cat "$SYS_DEV/id/vendor")
PRODUCT=$(cat "$SYS_DEV/id/product")
DEV_NAME=$(cat "$SYS_DEV/name")

echo "Device  : $DEV_NAME"
echo "Vendor  : $VENDOR  Product : $PRODUCT"

cat > "$RULE_FILE" <<EOF
# $DEV_NAME
SUBSYSTEM=="input", ATTRS{idVendor}=="$VENDOR", ATTRS{idProduct}=="$PRODUCT", GROUP="input", MODE="0664"
EOF

echo "Written : $RULE_FILE"

udevadm control --reload-rules
udevadm trigger

echo ""
echo "Done. Unplug and replug the footpedal — then run without sudo:"
echo "  python scripts/test_footpedal.py"
