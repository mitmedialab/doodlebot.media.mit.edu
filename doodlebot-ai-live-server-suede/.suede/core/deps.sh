#!/usr/bin/env bash
# .suede/core/deps.sh — ships INSIDE a suede dependency:
#     <repo>/<dependency>/.suede/core/deps.sh
#
# What THIS dependency needs beside it, and the exact commands that would put
# it there:
#
#     bash <dependency>/.suede/core/deps.sh            # the recipe
#     bash <dependency>/.suede/core/deps.sh --check    # exit 1 if anything is unresolved
#     bash <dependency>/.suede/core/deps.sh --in <dir> # act on the dependency at <dir>
#     bash <dependency>/.suede/core/deps.sh --https    # skip SSH: no key here
#
# --https reaches remotes over HTTPS only, and is passed on to every install
# command the recipe prints, so the whole recipe makes no SSH attempt.
#
# A dependency publishes its own dependencies as `.suede/.dependencies/
# <sibling>.gitrepo` records. Each one names a folder it expects to find
# NEXT TO ITSELF - `<dependency>/../<sibling>` - and the remote and commit that
# folder should hold. This script reads those records, looks at the disk, and
# for each one tells you one of:
#
#   satisfied    the sibling is there and points at the right repository
#   reuse        it is missing, but the same repository is already installed
#                somewhere in this repo at the same commit with no local
#                changes - one `ln -s` finishes it
#   decide       it is missing, and what is installed differs from what is
#                asked for - you choose between linking to what you have and
#                installing the exact commit under another name
#   install      nothing is installed - an install plus an `ln -s`
#
# It recurses: a dependency it would have you install has dependencies of its
# own, and you are told about all of them up front rather than one install at
# a time. Every command it prints is meant to be run from the repository root,
# and nothing is done for you - this script only reads.
#
# --check skips the network (no recursion into uninstalled dependencies, no
# local-change comparison) and exits 1 when any record is not satisfied - and,
# unlike the recipe, a sibling at a different commit than its record asks for
# is not satisfied: a release ships its records, so what it was built against
# has to be exactly what they name, all the way down. That is what the publish
# guard runs.
#
# Needs `git`. The reuse check runs the `diff` script beside this one.
#
# Env:
#   SUEDE_INSTALL_URL   what the recipe tells you to run (default
#                       https://suede.sh/install/release)

set -euo pipefail

usage() { grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'; exit 0; }
die() { printf 'deps: %s\n' "$*" >&2; exit 2; }
note() { printf 'deps: %s\n' "$*" >&2; }

INSTALL_URL="${SUEDE_INSTALL_URL:-https://suede.sh/install/release}"
RELEASE_DIR="release"

CHECK=0
HTTPS_ONLY=0
IN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    --check)   CHECK=1; shift ;;
    --in)      [[ -n "${2-}" ]] || die "--in needs a directory"; IN="$2"; shift 2 ;;
    --https)   HTTPS_ONLY=1; shift ;;
    *)         die "unknown argument: $1" ;;
  esac
done

command -v git >/dev/null 2>&1 || die "git not found"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)" \
  || die "cannot resolve script location"
DIFF="$SELF_DIR/diff"

if [[ -n "$IN" ]]; then
  DEP_DIR="$(cd "$IN" >/dev/null 2>&1 && pwd -P)" || die "no such directory: $IN"
else
  # Exactly two levels up (<dep>/.suede/core/deps.sh); `pwd -P` above already
  # resolved whatever symlink the caller came through.
  DEP_DIR="$(cd "$SELF_DIR/../.." >/dev/null 2>&1 && pwd -P)" \
    || die "cannot resolve dependency root above $SELF_DIR"
  [[ -f "$DEP_DIR/.gitrepo" ]] \
    || die "no .gitrepo at $DEP_DIR - is this inside an installed suede dependency?"
fi
# Fetched and run as `bash <(curl ...)`, there is no diff beside this script;
# the dependency's own copy will do.
[[ -f "$DIFF" ]] || DIFF="$DEP_DIR/.suede/core/diff"

ROOT="$(git -C "$DEP_DIR" rev-parse --show-toplevel 2>/dev/null)" \
  || die "$DEP_DIR is not inside a git repository"
ROOT="$(cd "$ROOT" && pwd -P)"

WORKSPACE="$(mktemp -d)"
trap 'rm -rf "$WORKSPACE"' EXIT

# --- small helpers ----------------------------------------------------------

# One spelling for one repository, so the ssh and https forms - and the local
# paths the tests use - compare equal.
identity() { # <url>
  local url="${1%/}"
  url="${url%.git}"
  url="${url#*://}"
  url="${url#*@}"
  printf '%s' "${url/:/\/}"
}

# The dependency's own name: the remote's basename, which is what the installer
# names the folder.
name_of() { # <url>
  local url="${1%/}"
  url="${url%.git}"
  printf '%s' "${url##*[:/]}"
}

short() { printf '%s' "${1:0:7}"; }

# <path> relative to <base>, both absolute; pure shell, since macOS has no
# `realpath --relative-to`.
relpath() { # <base> <path>
  local base="${1%/}" path="${2%/}" up="" rest
  [[ "$path" == "$base" ]] && { printf '.'; return; }
  while [[ "$path" != "$base"/* && "$base" != "/" && -n "$base" ]]; do
    base="$(dirname "$base")"; up="../$up"
  done
  rest="${path#"$base"}"; rest="${rest#/}"
  [[ "$base" == "/" ]] && rest="${path#/}"
  printf '%s%s' "$up" "$rest"
}

# Relative to the repository root, which is where every printed command runs.
from_root() { relpath "$ROOT" "$1"; }

field() { git config -f "$1" --get "subrepo.$2" 2>/dev/null || true; }

# A vendored dependent (one under release/) ships whole, so its siblings have to
# be inside release/ too: a link out of release/ reaches consumers broken.
in_release() { # <absolute path>
  [[ "$1" == "$ROOT/$RELEASE_DIR" || "$1" == "$ROOT/$RELEASE_DIR"/* ]]
}

# --- what is on disk --------------------------------------------------------

# Every subrepo in the repository, as "<dir>\t<identity>\t<commit>" lines, read
# once. Nested checkouts and node_modules are skipped; suede's own vendored
# machinery is harmless here because no record ever asks for it.
INSTALLED=""
scan_installed() {
  local file dir
  while IFS= read -r file; do
    dir="$(dirname "$file")"
    INSTALLED+="$dir	$(identity "$(field "$file" remote)")	$(field "$file" commit)"$'\n'
  done < <(find "$ROOT" -name .gitrepo -not -path '*/.git/*' -not -path '*/node_modules/*' \
             -not -path '*/.worktrees/*' | sort)
}

# Candidate installs for a record: same repository, inside release/ when the
# dependent is. Prints "<dir>\t<commit>" lines, exact-commit matches first.
candidates() { # <identity> <scope-dir> <wanted commit>
  local wanted="$1" scope="$2" line dir ident commit matches="" others=""
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    IFS=$'\t' read -r dir ident commit <<<"$line"
    [[ "$ident" == "$wanted" ]] || continue
    if in_release "$scope"; then in_release "$dir" || continue; fi
    [[ "$dir" == "$DEP_DIR" ]] && continue
    if [[ "$commit" == "$3" ]]; then matches+="$dir	$commit"$'\n'; else others+="$dir	$commit"$'\n'; fi
  done <<<"$INSTALLED"
  printf '%s%s' "$matches" "$others"
}

# Installs the recipe has already told you to make, as
# "<identity>\t<commit>\t<dir>\t<label>" lines, so a second dependent wanting
# the same pin gets one `ln -s` instead of a second install.
PLANNED=""
planned_for() { # <identity> <commit>
  local line ident commit dir label
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    IFS=$'\t' read -r ident commit dir label <<<"$line"
    [[ "$ident" == "$1" && "$commit" == "$2" ]] && { printf '%s\t%s' "$dir" "$label"; return 0; }
  done <<<"$PLANNED"
  return 1
}

# Directories (real or planned) already walked, so a cycle or a diamond is
# reported once.
VISITED=""
visited() { grep -qxF -- "$1" <<<"$VISITED"; }

# --- the other side of the network ------------------------------------------

# The ways to reach one repository, the recorded one first. An installed
# dependency records the SSH spelling, because that is the one `upstream` can
# push through; a machine with no key - a CI runner, a fresh container - reaches
# the same public repository over HTTPS. So fetch from what is recorded, and
# fall back to the other spelling. A local path, or an address carrying a port,
# has one spelling: itself. With --https (HTTPS_ONLY=1) only the HTTPS spelling
# is tried: you know there is no SSH key here, so do not spend a timeout
# finding out.
spellings() { # <url>
  local url="$1" rest host path
  case "$url" in
    https://*|http://*) rest="${url#*://}"; rest="${rest#*@}"; host="${rest%%/*}"; path="${rest#*/}" ;;
    ssh://*)            rest="${url#ssh://}"; rest="${rest#*@}"; host="${rest%%/*}"; path="${rest#*/}" ;;
    *@*:*)              rest="${url#*@}"; host="${rest%%:*}"; path="${rest#*:}" ;;
    *)                  printf '%s\n' "$url"; return 0 ;;
  esac
  if [[ -z "$host" || -z "$path" || "$host" == *:* ]]; then printf '%s\n' "$url"; return 0; fi
  path="${path%/}"; path="${path%.git}"
  case "$url" in
    https://*|http://*)
      printf '%s\n' "$url"
      [[ "${HTTPS_ONLY:-0}" == 1 ]] || printf 'git@%s:%s.git\n' "$host" "$path" ;;
    *)
      [[ "${HTTPS_ONLY:-0}" == 1 ]] || printf '%s\n' "$url"
      printf 'https://%s/%s.git\n' "$host" "$path" ;;
  esac
}

# Our own fetches fail fast and never ask anyone anything: no password prompt,
# no host-key question, five seconds to connect.
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes -o ConnectTimeout=5}"
export GIT_TERMINAL_PROMPT=0

# The manifest of a dependency that is not installed, fetched at the commit a
# record asks for. Only `.suede/.dependencies/*.gitrepo` is materialised.
fetch_manifest() { # <remote> <commit> <branch> <destination>
  local remote="$1" commit="$2" branch="${3:-release}" dest="$4" scratch name url reached=""
  scratch="$WORKSPACE/fetch-$$-$RANDOM"
  mkdir -p "$dest"
  git init --quiet "$scratch"
  while IFS= read -r url; do
    git -C "$scratch" remote remove origin 2>/dev/null || true
    git -C "$scratch" remote add origin "$url"
    git -C "$scratch" fetch --quiet --depth 1 origin "$commit" 2>/dev/null \
      || git -C "$scratch" fetch --quiet origin "refs/heads/$branch" 2>/dev/null \
      || continue
    reached="$url"; break
  done < <(spellings "$remote")
  [[ -n "$reached" ]] || { rm -rf "$scratch"; return 1; }
  git -C "$scratch" cat-file -e "$commit^{commit}" 2>/dev/null || { rm -rf "$scratch"; return 1; }
  while IFS= read -r name; do
    [[ "$name" == *.gitrepo ]] || continue
    git -C "$scratch" show "$commit:$name" > "$dest/$(basename "$name")"
  done < <(git -C "$scratch" ls-tree --name-only "$commit" ".suede/.dependencies/" 2>/dev/null || true)
  rm -rf "$scratch"
}

# Does an install at the asked-for commit also match it byte for byte? Exit 0
# yes, 1 no (local changes), 2 could not tell.
unchanged_at() { # <dir> <commit>
  local https=()
  [[ "$HTTPS_ONLY" == 1 ]] && https=(--https)
  bash "$DIFF" --in "$1" --at "$2" --quiet ${https[@]+"${https[@]}"} >/dev/null 2>&1
}

# --- output -----------------------------------------------------------------

UNRESOLVED=0
SATISFIED=0
MISMATCHED=0
INDENT=""

say() { if [[ -z "$*" ]]; then echo; else printf '%s%s\n' "$INDENT" "$*"; fi; }
cmd() { printf '%s    %s\n' "$INDENT" "$*"; }

# `ln -s` from the root: the link's path is root-relative, its target is
# relative to the directory the link sits in.
link_command() { # <link absolute> <target absolute>
  printf 'ln -s %s %s' "$(relpath "$(dirname "$1")" "$2")" "$(from_root "$1")"
}

# An install runs in the dependent's directory so the new folder lands beside
# it; from the root that is a subshell with a cd, or nothing when the
# dependent is at the root already.
#
# --transitive on every one: what a recipe installs is there for a dependency's
# edge, not because this repository's own code imports it. The ln -s printed
# beside it is the link that matters; the installer adds no declaration.
install_command() { # <parent absolute> <remote> <commit> [extra flags]
  local parent="$1" remote="$2" commit="$3" extra="${4-} --transitive" where
  [[ "$HTTPS_ONLY" == 1 ]] && extra="$extra --https"
  where="$(from_root "$parent")"
  if [[ "$where" == "." ]]; then
    printf 'bash <(curl -fsSL %s) --repo %s --at %s%s' "$INSTALL_URL" "$remote" "$commit" "$extra"
  else
    printf '(cd %s && bash <(curl -fsSL %s) --repo %s --at %s%s)' "$where" "$INSTALL_URL" "$remote" "$commit" "$extra"
  fi
}

# --- the walk ---------------------------------------------------------------

# <manifest dir> is where the records are read from (on disk, or fetched);
# <dependent path> is where the dependent is or will be, which fixes where its
# siblings go; <label prefix> numbers the output.
walk() { # <manifest dir> <dependent path> <label prefix>
  local manifest="$1" dependent="$2" prefix="$3"
  local parent record entry remote commit branch sibling index=0 label

  parent="$(dirname "$dependent")"
  local records=()
  if [[ -d "$manifest" ]]; then
    for record in "$manifest"/*.gitrepo; do [[ -e "$record" ]] && records+=("$record"); done
  fi

  if [[ -z "$prefix" ]]; then
    local where; where="$(from_root "$parent")"
    [[ "$where" == "." ]] && where="at the repository root" || where="in $where/"
    say "deps: $(basename "$dependent") needs ${#records[@]} sibling(s)$( [[ ${#records[@]} -gt 0 ]] && printf ' %s' "$where" )"
    [[ ${#records[@]} -gt 0 ]] && say ""
  fi

  for record in ${records[@]+"${records[@]}"}; do
    index=$((index + 1))
    label="${prefix}${index}"
    entry="$(basename "$record" .gitrepo)"
    remote="$(field "$record" remote)"
    commit="$(field "$record" commit)"
    branch="$(field "$record" branch)"
    [[ -n "$remote" && -n "$commit" ]] || { say "[$label] $entry: record names no remote or commit; skipping"; continue; }
    sibling="$parent/$entry"
    one_record "$label" "$entry" "$sibling" "$remote" "$commit" "${branch:-release}"
  done
}

one_record() { # <label> <entry> <sibling abs> <remote> <commit> <branch>
  local label="$1" entry="$2" sibling="$3" remote="$4" commit="$5" branch="$6"
  local ident parent name have_remote have_commit real planned line dir cand cand_commit

  ident="$(identity "$remote")"
  parent="$(dirname "$sibling")"
  name="$(name_of "$remote")"

  say "[$label] $entry"
  say "    $remote @ $(short "$commit")"

  if [[ -L "$sibling" && ! -e "$sibling" ]]; then
    UNRESOLVED=$((UNRESOLVED + 1))
    say "    $(from_root "$sibling") is a dangling symlink -> $(readlink "$sibling")"
    say "    remove it, then:"
    cmd "rm $(from_root "$sibling")"
    resolve_missing "$label" "$entry" "$sibling" "$remote" "$commit" "$branch" "$ident" "$name"
    return
  fi

  if [[ -e "$sibling" ]]; then
    if [[ ! -f "$sibling/.gitrepo" ]]; then
      UNRESOLVED=$((UNRESOLVED + 1))
      say "    $(from_root "$sibling") exists but is not a suede install (no .gitrepo)"
      say "    move it out of the way, then install:"
      cmd "$(install_command "$parent" "$remote" "$commit")"
      cmd "$(link_command "$sibling" "$parent/$name")"
      say ""
      return
    fi
    have_remote="$(field "$sibling/.gitrepo" remote)"
    have_commit="$(field "$sibling/.gitrepo" commit)"
    real="$(cd "$sibling" && pwd -P)"
    if [[ "$(identity "$have_remote")" != "$ident" ]]; then
      UNRESOLVED=$((UNRESOLVED + 1))
      say "    $(from_root "$sibling") -> $(from_root "$real") points at a different repository: $have_remote"
      say "    repoint it, or remove it and install what is asked for:"
      cmd "rm $(from_root "$sibling")"
      cmd "$(install_command "$parent" "$remote" "$commit" " --name $name-$(short "$commit")")"
      cmd "$(link_command "$sibling" "$parent/$name-$(short "$commit")")"
      say ""
      return
    fi
    if in_release "$parent" && ! in_release "$real"; then
      UNRESOLVED=$((UNRESOLVED + 1))
      say "    $(from_root "$sibling") -> $(from_root "$real") points outside $RELEASE_DIR/"
      say "    a vendored dependency ships whole, so its siblings have to ship with it; vendor a copy beside it:"
      cmd "rm $(from_root "$sibling")"
      cmd "$(install_command "$parent" "$remote" "$commit")"
      cmd "$(link_command "$sibling" "$parent/$name")"
      say ""
      return
    fi
    if [[ "$have_commit" == "$commit" ]]; then
      SATISFIED=$((SATISFIED + 1))
      say "    satisfied by $(from_root "$real") @ $(short "$have_commit"), matches the pin"
    elif [[ "$CHECK" == 1 ]]; then
      # What a release ships is its records. Resolved here to another commit,
      # this tree was built against something its consumers will never install.
      UNRESOLVED=$((UNRESOLVED + 1))
      say "    resolves to $(from_root "$real") @ $(short "$have_commit"), but the record asks for $(short "$commit")"
      say "    a release cannot ship that: its consumers will install $(short "$commit"), not what was built against"
    else
      SATISFIED=$((SATISFIED + 1))
      MISMATCHED=$((MISMATCHED + 1))
      say "    satisfied by $(from_root "$real") @ $(short "$have_commit") - NOT the $(short "$commit") that $entry asks for"
      say "    fine while you work, but push-release will refuse to publish it. See what differs between"
      say "    the commit $entry was built against and what you have:"
      cmd "bash $(from_root "$real")/.suede/core/diff --at $commit"
      say "    then either sync the dependent to a release built against $(short "$have_commit"), or install"
      say "    exactly what it asks for beside yours and point the edge there:"
      cmd "$(install_command "$parent" "$remote" "$commit" " --name $name-$(short "$commit")")"
      cmd "rm $(from_root "$sibling")"
      cmd "$(link_command "$sibling" "$parent/$name-$(short "$commit")")"
    fi
    say ""
    if ! visited "$real"; then
      VISITED+="$real"$'\n'
      INDENT="$INDENT    "
      walk "$real/.suede/.dependencies" "$real" "$label."
      INDENT="${INDENT%    }"
    fi
    return
  fi

  resolve_missing "$label" "$entry" "$sibling" "$remote" "$commit" "$branch" "$ident" "$name"
}

resolve_missing() { # <label> <entry> <sibling abs> <remote> <commit> <branch> <identity> <name>
  local label="$1" entry="$2" sibling="$3" remote="$4" commit="$5" branch="$6" ident="$7" name="$8"
  local parent planned dir plabel line cand cand_commit install_name target

  parent="$(dirname "$sibling")"
  UNRESOLVED=$((UNRESOLVED + 1))

  # Something earlier in this recipe already installs exactly this.
  if planned="$(planned_for "$ident" "$commit")"; then
    IFS=$'\t' read -r dir plabel <<<"$planned"
    say "    same install as [$plabel]"
    cmd "$(link_command "$sibling" "$dir")"
    say ""
    return
  fi

  # Something already on disk might do.
  local found=""
  found="$(candidates "$ident" "$parent" "$commit")"
  if [[ -n "$found" ]]; then
    line="$(printf '%s' "$found" | head -1)"
    IFS=$'\t' read -r cand cand_commit <<<"$line"
    if [[ "$cand_commit" == "$commit" && "$CHECK" == 0 ]] && unchanged_at "$cand" "$commit"; then
      say "    found $(from_root "$cand") at the same commit with no local changes"
      cmd "$(link_command "$sibling" "$cand")"
      say ""
      PLANNED+="$ident	$commit	$cand	$label"$'\n'
      if ! visited "$cand"; then
        VISITED+="$cand"$'\n'
        INDENT="$INDENT    "
        walk "$cand/.suede/.dependencies" "$cand" "$label."
        INDENT="${INDENT%    }"
      fi
      return
    fi
    if [[ "$CHECK" == 1 ]]; then
      say "    missing; $(from_root "$cand") @ $(short "$cand_commit") is from the same repository"
      say ""
      return
    fi
    install_name="$name-$(short "$commit")"
    if [[ "$cand_commit" == "$commit" ]]; then
      say "    found $(from_root "$cand") at the same commit, but with local changes"
    else
      say "    found $(from_root "$cand") from the same repository, but at $(short "$cand_commit")"
    fi
    say "    see what differs between what $entry asks for and what you have:"
    cmd "bash $(from_root "$cand")/.suede/core/diff --at $commit"
    say "    then either point the edge at your copy and own the difference:"
    cmd "$(link_command "$sibling" "$cand")"
    say "    or install exactly what it asked for beside it, under another name:"
    cmd "$(install_command "$parent" "$remote" "$commit" " --name $install_name")"
    cmd "$(link_command "$sibling" "$parent/$install_name")"
    say "    (resolve this one and re-run; its own dependencies depend on which you choose)"
    say ""
    return
  fi

  # Nothing anywhere: install it beside the dependent.
  target="$parent/$name"
  install_name=""
  if [[ -e "$target" || -L "$target" ]]; then
    # The name is taken by something that is not this repository (or a
    # candidate would have matched), so give the install another one.
    install_name="$name-$(short "$commit")"
    target="$parent/$install_name"
  fi
  if [[ "$CHECK" == 1 ]]; then
    say "    missing; not installed anywhere in this repository"
    say ""
    return
  fi
  say "    not installed anywhere in this repository"
  cmd "$(install_command "$parent" "$remote" "$commit" "${install_name:+ --name $install_name}")"
  cmd "$(link_command "$sibling" "$target")"
  say ""
  PLANNED+="$ident	$commit	$target	$label"$'\n'

  # Look ahead: what will that install need beside it?
  local key="$ident@$commit" fetched
  visited "$key" && return
  VISITED+="$key"$'\n'
  fetched="$WORKSPACE/manifest-$RANDOM$RANDOM"
  note "fetching the dependency list of $name @ $(short "$commit")"
  if fetch_manifest "$remote" "$commit" "$branch" "$fetched"; then
    INDENT="$INDENT    "
    walk "$fetched" "$target" "$label."
    INDENT="${INDENT%    }"
  else
    say "    (could not reach $remote to read its own dependencies; run deps.sh after installing it)"
    say ""
  fi
}

scan_installed
VISITED+="$DEP_DIR"$'\n'
walk "$DEP_DIR/.suede/.dependencies" "$DEP_DIR" ""

mismatch_note() {
  [[ "$MISMATCHED" -gt 0 ]] || return 0
  say "deps: $MISMATCHED of them at a different commit than asked for - fine while you work, but push-release"
  say "deps: refuses to publish until they match; the commands above show the difference and the fix"
}

if [[ "$UNRESOLVED" == 0 ]]; then
  say "deps: everything is in place ($SATISFIED satisfied)"
  mismatch_note
  exit 0
fi

say "deps: $SATISFIED satisfied, $UNRESOLVED to resolve."
mismatch_note
if [[ "$CHECK" == 1 ]]; then
  say "deps: run without --check for the commands that would resolve them"
  exit 1
fi
say "deps: run the commands above from the repository root: $ROOT"
say "deps: then re-run this script to confirm. Drop --at to take a release tip instead of the pinned commit."
exit 0
