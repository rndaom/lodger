## What and why

<!-- One change per pull request. What does it do, and why? -->

## How I tested it

<!-- Commands, and on which hardware (card, driver, distro). Tests need no GPU; say if you also ran on a card. -->

## Checklist

- [ ] `ruff check .` and `python -m unittest discover tests` pass
- [ ] New behaviour has a test, and a line in CHANGELOG.md under "Unreleased"
- [ ] If this touches the graphics card: I followed the [safety rules](https://github.com/rndaom/lodger/blob/main/CONTRIBUTING.md#safety-rules-for-contributors) (2 GiB floor kept, tested on the processor first)
- [ ] Comments are in plain words and match the code around them
