# Security

## Reporting a vulnerability

Please report security problems **privately**, not in a public issue.

Open this repository's **Security** tab, choose **Report a vulnerability**, and describe what you found, how to
reproduce it, and what you think the impact is. Only the maintainers can see the report. You will get a reply as
soon as the maintainer can manage; lodger is a small volunteer project, so there is no guaranteed turnaround, but
reports are taken seriously and fixes for real problems come first. Credit in the release notes is given unless
you would rather stay unnamed.

## Supported versions

lodger is pre-1.0. Only the **latest release** gets security fixes.

| Version | Supported |
|---|---|
| latest 0.x release | yes |
| older releases | no, please upgrade |

## The localhost-only stance

lodger is meant to run on your own PC, for apps on that same PC.

- The server binds to **127.0.0.1 only**. There is no option to listen on other addresses, and there should
  not be one. Do not put lodger behind a reverse proxy or port forward on the open internet.
- Today there is **no API key**: anything running on this PC as any user can talk to it, and a web page open in
  your browser could try to. Protections against the browser case (a check of the `Host` and `Origin` headers),
  an optional API key, and a key for the engine lodger starts are being added. Until they land, treat the server
  as reachable by every program on the machine.
- The llama.cpp engine lodger starts listens on a random private port on 127.0.0.1 only.
- lodger does not send your prompts or answers anywhere. Requests stay on your machine.

## What counts as a vulnerability

Examples worth reporting:

- A web page, or another user on the machine, getting the server to do something it should not (DNS rebinding,
  cross-site requests, reading other users' prompts).
- The server listening on an address other than 127.0.0.1.
- A crafted model name, request, `models.toml` or model file making lodger run commands, or read or write files,
  that it should not.
- Ways around the fence (RAM cap, no swap, low priority) that a request can trigger.

Not vulnerabilities in lodger itself (report them to the project that owns them):

- Bugs in llama.cpp, graphics drivers, systemd or the models you run.
- Anything that needs you to expose the server to a network on purpose.
- A model saying something you dislike.

A bug that makes lodger take the graphics card or RAM from your apps is not a security issue, but it is the
most serious kind of bug the project has: please file it as a normal bug report.
