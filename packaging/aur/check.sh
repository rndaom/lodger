#!/bin/bash
# Build, test and package lodger-git from this checkout in an Arch Linux container (CI runs it as root in
# archlinux:base-devel). Local run: docker run --rm -v "$PWD:/src:ro" archlinux:base-devel /src/packaging/aur/check.sh
set -euo pipefail
src=${1:-/src}
pacman -Syu --noconfirm --needed git python python-build python-installer python-setuptools python-wheel \
  python-numpy python-yaml python-tqdm python-requests >/dev/null
useradd -m builder
# python-gguf is in the AUR; gguf from PyPI stands in for it here (with makepkg -d)
su builder -c "python -m pip install --user --break-system-packages --quiet gguf" 2>/dev/null \
  || { pacman -S --noconfirm --needed python-pip >/dev/null; su builder -c "python -m pip install --user --break-system-packages --quiet gguf"; }
git config --global --add safe.directory "*"
git clone --quiet "$src" /home/builder/lodger
cd /home/builder/lodger
commit=$(git rev-parse HEAD)
mkdir /home/builder/pkg
sed "s|^source=.*|source=(\"git+file:///home/builder/lodger#commit=$commit\")|" packaging/aur/lodger-git/PKGBUILD \
  > /home/builder/pkg/PKGBUILD
chown -R builder: /home/builder
cd /home/builder/pkg
su builder -c "makepkg -d --noconfirm"
pkg=$(ls lodger-git-*.pkg.tar.zst)
pacman -U --noconfirm -dd "$pkg" >/dev/null
lodger --version
lodger --help >/dev/null
su builder -c "lodger doctor" >/dev/null
grep -q '^ExecStart=/usr/bin/lodger serve' /usr/lib/systemd/user/lodger.service
echo "lodger-git: built, tested, installed and ran ($pkg)"
