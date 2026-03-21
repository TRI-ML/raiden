#!/usr/bin/env bash
# Install a udev rule for 3Dconnexion SpaceMice so they are accessible without sudo.
# Run once: sudo bash scripts/install_spacemouse_udev.sh
# Then unplug and replug the SpaceMouse.

set -euo pipefail

RULE_FILE=/etc/udev/rules.d/99-spacemouse.rules

cat > "$RULE_FILE" <<'RULES'
# 3Dconnexion SpaceMouse devices
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", GROUP="plugdev", MODE="0664"
RULES

echo "Written : $RULE_FILE"

udevadm control --reload-rules
udevadm trigger

echo ""
echo "Done. Make sure your user is in the 'plugdev' group:"
echo "  sudo usermod -aG plugdev \$USER"
echo "Then log out and back in, and replug the SpaceMouse."
