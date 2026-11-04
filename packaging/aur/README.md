# Arch User Repository

`lodger-git/` builds lodger from the newest commit on GitHub: `python -m build` into a wheel, installed into
the system Python, with the unit test suite as `check()`. It also installs the user unit as
`/usr/lib/systemd/user/lodger.service` (running `/usr/bin/lodger`), so after installing:

```bash
systemctl --user enable --now lodger
```

(`lodger service install` does the same: it sees the packaged unit and only enables it.)

It needs only Python: lodger reads model files itself. A plain `lodger` package (from release tarballs) follows the first tagged
release.

## Publishing (maintainer)

```bash
cd packaging/aur/lodger-git
makepkg -f && makepkg --printsrcinfo > .SRCINFO   # builds, tests, updates pkgver
git clone ssh://aur@aur.archlinux.org/lodger-git.git /tmp/aur-lodger-git
cp PKGBUILD .SRCINFO /tmp/aur-lodger-git/ && cd /tmp/aur-lodger-git
git add PKGBUILD .SRCINFO && git commit -m "Update to $(sed -n 's/^\tpkgver = //p' .SRCINFO)" && git push
```
