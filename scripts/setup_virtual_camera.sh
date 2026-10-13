#!/usr/bin/env bash
# Linux: create a v4l2loopback device that conferencing apps list as "Privacy Lens".
# Needs: sudo pacman -S v4l2loopback-dkms linux-headers   (Arch/Omarchy)   |   sudo apt install v4l2loopback-dkms
set -euo pipefail
if ! modinfo v4l2loopback >/dev/null 2>&1; then
  echo "v4l2loopback is not installed. On Arch: sudo pacman -S --needed v4l2loopback-dkms linux-headers" >&2; exit 1
fi
sudo modprobe -r v4l2loopback 2>/dev/null || true
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Privacy Lens" exclusive_caps=1
v4l2-ctl --list-devices | grep -A1 "Privacy Lens"
echo "Done. Start the app: python -m app.main  (optionally set output.device: /dev/video10 in config/default.yaml)"
