#!/usr/bin/env bash
# Install a udev rule for 3Dconnexion SpaceMouse devices so they are
# accessible without sudo.
#
# Run once: sudo bash scripts/install_spacemouse_udev.sh
# Then log out and back in (or reboot) for group membership to take effect.
# USB devices also need to be unplugged and replugged.

set -euo pipefail

RULE_FILE=/etc/udev/rules.d/99-spacemouse.rules
VENDOR_ID="256f"  # 3Dconnexion

cat > "$RULE_FILE" <<EOF
# 3Dconnexion SpaceMouse — grant read/write access without sudo
KERNEL=="hidraw*", ATTRS{idVendor}=="$VENDOR_ID", MODE="0666"
EOF

echo "Written : $RULE_FILE"

udevadm control --reload-rules
udevadm trigger

# Add current user to the input group (takes effect after re-login)
if ! groups "$USER" | grep -q '\binput\b'; then
    usermod -a -G input "$USER"
    echo "Added $USER to the 'input' group."
    echo "Log out and back in (or reboot) for the group change to take effect."
else
    echo "$USER is already in the 'input' group."
fi

echo ""
echo "Done. Verify with: ls -la /dev/hidraw*"
