# [Suede](https://github.com/pmalacho-mit/suede) Core (`release`)

Vendored at `.suede/core` on your dependency's **`release`** branch — which
means it ships, and a consumer who installs your dependency finds these at
`<dependency>/.suede/core/`. They are the tools for working with an installed
dependency. All of them are bash, written for the bash macOS ships (3.2), and
none of them takes a target: each acts on the dependency it lives inside.

The maintainer's half (the publish guard, `extract`, `list`, the project-wide
divergence audit) is vendored from `dependency/main/core` onto `main`, at the
same path.

This folder is a [git-subrepo](https://github.com/ingydotnet/git-subrepo) of the
suede library, so you get fixes by pulling. **From `main`**, where it lives at
`release/.suede/core` — like everything else that ships, it reaches the
`release` branch through `main`, and there is never a reason to check `release`
out:

```bash
bash .suede/core/sync.sh          # updates this and every other suede subrepo
```

(`sync.sh` also clears the two leftovers a pull of a subrepo nested under
`release/` leaves behind, which would otherwise stop the next publish.)

## [deps.sh](./deps.sh)

What **this dependency needs beside it**, and the exact commands that would
put it there.

```bash
bash <dependency>/.suede/core/deps.sh            # the recipe
bash <dependency>/.suede/core/deps.sh --check    # exit 1 if anything is unresolved
bash <dependency>/.suede/core/deps.sh --in <dir> # act on the dependency at <dir>
```

A dependency publishes its own dependencies as records in
`.suede/.dependencies/`: one `<sibling>.gitrepo` per dependency, naming the
folder it expects to find **next to itself** (`<dependency>/../<sibling>`) and
the remote and commit that folder should hold. `deps.sh` reads those records,
looks at the disk, and for each one tells you one of:

| Outcome | Meaning | What it prints |
| --- | --- | --- |
| **satisfied** | the sibling is there and points at the right repository | the commit, and whether it matches the pin |
| **reuse** | missing, but the same repository is already installed somewhere in your repo at the same commit with no local changes | one `ln -s` |
| **decide** | missing, and what you have installed differs from what is asked for | the `diff --at` that shows the difference, then both options: link to what you have, or install the exact commit under another name |
| **install** | nothing is installed | the install command and the `ln -s` |

It **recurses**: a dependency it would have you install has dependencies of its
own, and you see all of them up front — numbered `[1]`, `[1.1]`, `[2]` — rather
than one install at a time. A pin two dependents share is installed once and
linked twice ("same install as [1.1]"). Every command is meant to be run from
your repository root, and **nothing is done for you**: the script only reads.
The installs it prints carry `--transitive`, so they add no declaration to your
repository: the `ln -s` beside each one is the edge that matters.
The installer runs it for you at the end of every install.

A sibling that resolves to the right repository at a *different* commit is
accepted by the recipe, so you can work against whatever you choose, but the
readout says so in so many words. It prints the `diff --at <commit>` that shows
what differs between the commit the dependency was built against and what you
have, and the fix: sync the dependent, or install the exact commit beside
yours and point the edge there.

`--check` is the publish guard's mode: no network, no recursion into
dependencies you have not installed, and exit `1` the moment a record is not
satisfied. It is strict where the recipe is lenient: a sibling at a different
commit than its record names fails, because a release ships its records and
its consumers will install exactly those commits.

A **vendored** dependency (one installed inside `release/`) is held to one more
rule: its siblings have to be inside `release/` too, because a link out of
`release/` reaches consumers broken. `deps.sh` places those installs inside
`release/` and refuses to offer a copy from outside it.

Needs `git`, and the [`diff`](#diff) beside it for the reuse check. Like
`diff`, it falls back from a record's remote to its other spelling (SSH ↔
HTTPS) when fetching what it looks ahead at. With `--https` it skips the SSH
attempt, and puts `--https` on every install command it prints.

## [clean](./clean)

`git subrepo clean` for **this** dependency, runnable from any working
directory and through any symlink to it.

```bash
bash <dependency>/.suede/core/clean            # the subrepo/<path> branch and scratch worktree
bash <dependency>/.suede/core/clean --force    # and the fetched refs too
```

Reach for it when a `sync` or `upstream` stopped part way: git-subrepo leaves a
`subrepo/<path>` branch and a worktree under `.git/tmp/subrepo/` behind, and
either can make the next pull or push refuse to start. It deletes only that
bookkeeping — never your files, never your commits. Anything you pass goes to
`git subrepo clean`; afterwards it also removes this dependency's scratch
directory if git-subrepo missed it, and prunes stale worktrees.

## [diff](./diff)

What your copy of this dependency differs from — in any of three directions.

```bash
bash <dependency>/.suede/core/diff                # what you would propose
bash <dependency>/.suede/core/diff --sync         # what you would receive
bash <dependency>/.suede/core/diff --https        # skip SSH: no key here
bash <dependency>/.suede/core/diff --at <commit>  # against some other commit
```

By default the pinned commit is on the left and your working tree on the right,
so `+` lines are yours: this is the change [`upstream`](#upstream) would
propose. With `--sync` your tree is on the left and the current tip of the
remote's `release` branch on the right, so `+` lines are incoming: this is what
[`sync`](#sync) would bring you. `--sync` also names the two commits, and says
so outright when your pin *is* the tip and a sync would bring nothing.

`--at <commit>` puts that commit on the left instead of your pin. This is how
you compare what you have against what *another* dependency asks for; `deps.sh`
prints the exact command when it finds such a case.

`--in <dir>` acts on the dependency at `<dir>` instead of the one containing the
script — the one exception to "no target", so that `deps.sh` and the publish
guard can reuse a single copy for every dependency in a tree.

Your side is the files as they are on disk: uncommitted edits and files you
have only just created are in, and anything your `.gitignore` excludes is out.
`.gitrepo` is dropped from both sides — it is local bookkeeping and always
differs.

Anything else you pass goes to `git diff`, so `--stat`, `--name-only`,
`--quiet` and `-w` work as usual. Exit codes are `0` for no difference and `1`
for a difference, as `git diff` reports them, with `2` reserved for "could not
run at all" so a caller can tell an answer from a failure.

Needs only `git` and the ability to reach the remote; git-subrepo is not
required. The recorded remote is tried first and its other spelling second
(SSH ↔ HTTPS), so a dependency installed with an SSH remote still compares on a
machine with no key — a CI runner, a fresh container. Only when neither answers
does it exit `2`, naming both spellings it tried. `--https` skips the SSH
attempt altogether, when you know there is no key to find.

## [sync](./sync)

`git subrepo pull` for **this** dependency, runnable from any working directory.

```bash
bash <dependency>/.suede/core/sync
```

Anything you pass is handed straight to `git subrepo pull`, so `sync --force`
and the rest of git-subrepo's options work as documented there.

Three things it does that a bare `git subrepo pull` will not: it runs from the
repository root with a root-relative path, so where you are does not matter;
it resolves symlinks to the real folder first, because `git subrepo pull` on a
symlink path fails outright — and the edge entries between your dependencies
are symlinks; and when the recorded remote does not answer it pulls over the
other spelling (SSH ↔ HTTPS) instead, says so, and puts the recorded remote back
in the `.gitrepo` afterwards, since git-subrepo would otherwise keep the
fallback. Pass your own `-r`/`--remote` to skip that, or `--https` to skip
the SSH attempt and pull over HTTPS directly.

`git subrepo pull` also requires a clean working tree, while installing does
not. If you have just installed something, commit before syncing.

## [upstream](./upstream)

Propose this dependency's **local changes back to the library**, as a reviewable
PR against the library's `main`.

```bash
bash <dependency>/.suede/core/upstream
```

First commit the changes you want to send — the working tree must be clean.
You can run it through any path that reaches the dependency, including its
declaring symlink (`bash suede.nests.sweater-vest/.suede/core/upstream`): like
`sync`, it resolves to the real folder first, because `git subrepo push` on a
symlink path fails.

1. Splits the dependency's local commits out via `git subrepo` and pushes them to
   a deterministic branch on the library's remote:
   `downstream/<owner>/<repo>-<your-commit>`.
2. The library's `suede-downstream-to-main` workflow rebuilds that branch as a
   `main`-shaped PR head and opens the pull request for the maintainers to test,
   fix, and merge.
3. Your local state is restored afterward, so a later `sync` stays safe. The
   `release` branch is **never** modified, so other consumers are unaffected.

When the dependency is on GitHub, `upstream` then **waits for the PR and prints
its link** — up to three minutes, since the workflow that opens it takes about a
minute. A draft PR means replaying the change onto the current release hit
conflicts, left as markers for the maintainers. "Nothing new to propose" means
the dependency's `main` already has the change, so no PR was opened. If the
workflow fails, it prints the run's link and exits `1`; if no PR appears in
time, the Actions page and `2`. `--no-wait` skips the wait. It uses GitHub's
public API and needs `jq`; a token in `GH_TOKEN` lifts the hourly request
limit.

Each commit becomes its own proposal; re-running on the same commit is a no-op.
Pass `-r`/`--remote <name>` to push to a remote other than the one tracked in
the dependency's `.gitrepo`. It is a thin bootstrapper: the real logic is hosted
at `https://suede.sh/upstream` (override with `SUEDE_UPSTREAM_URL`). Requires
`git`, `curl` and git-subrepo.

## If `git subrepo` is not found

`sync` and `upstream` need it, and both look in one more place before giving
up: if `GIT_SUBREPO_ROOT` is set, they source `$GIT_SUBREPO_ROOT/.rc` to bring
the command into scope. That covers an install — a devcontainer feature, a
login shell profile — whose `PATH` a non-interactive script never inherited.
