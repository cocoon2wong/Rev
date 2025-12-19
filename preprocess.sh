#!/bin/sh
set -eu

REPO_URL="https://github.com/cocoon2wong/Project-Zero-Divided.git"
REPO_DIR="Project-Zero-Divided"
BRANCH="Unpredictable"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] required command not found: $1"
    exit 1
  fi
}

info() { printf '%s\n' "[INFO] $*"; }
warn() { printf '%s\n' "[WARN] $*"; }
err()  { printf '%s\n' "[ERROR] $*"; }

need_cmd git

HAS_RSYNC=0
if command -v rsync >/dev/null 2>&1; then
  HAS_RSYNC=1
fi

copy_tree() {
  src="$1"
  dst="$2"

  if [ ! -e "$src" ]; then
    warn "Missing in repo, skip: $src"
    return 0
  fi

  if [ -d "$dst" ]; then
    info "Update: $dst  <=  $src"
  else
    info "Create: $dst  <=  $src"
  fi

  if [ "$HAS_RSYNC" -eq 1 ]; then
    if [ -d "$src" ]; then
      mkdir -p "$dst"
      rsync -a "$src"/ "$dst"/
    else
      mkdir -p "$(dirname "$dst")"
      rsync -a "$src" "$dst"
    fi
  else
    if [ -d "$src" ]; then
      mkdir -p "$dst"
      cp -fR "$src"/. "$dst"/
    else
      mkdir -p "$(dirname "$dst")"
      cp -f "$src" "$dst"
    fi
  fi
}

sync_dir_overwrite() {
  src="$1"
  dst="$2"

  if [ ! -d "$src" ]; then
    info "No local override folder, skip: $src"
    return 0
  fi
  if [ ! -d "$dst" ]; then
    err "Destination folder must exist (from repo step): $dst"
    exit 1
  fi

  info "Apply local overrides: $src  ->  $dst (overwrite on conflict)"
  if [ "$HAS_RSYNC" -eq 1 ]; then
    rsync -a "$src"/ "$dst"/
  else
    cp -fR "$src"/. "$dst"/
  fi
}

info "Step 0/3: Repo presence check"

if [ -d "$REPO_DIR" ]; then
  info "Repo folder already exists, skip clone/checkout: $REPO_DIR"
else
  info "Step 1/3: Fetch theme/source repo"
  info "Cloning: $REPO_URL -> $REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"

  info "Checking out branch: $BRANCH"
  (
    cd "$REPO_DIR"
    git fetch --all --prune
    git checkout "$BRANCH"
  )
fi

info "Step 2/3: Copy base folders from repo into current project"

# Minimal safety: ensure expected repo folders exist
[ -d "$REPO_DIR" ] || { err "Repo folder not found: $REPO_DIR"; exit 1; }

copy_tree "./$REPO_DIR/_data"     "./_data"
copy_tree "./$REPO_DIR/_includes" "./_includes"
copy_tree "./$REPO_DIR/_layouts"  "./_layouts"
copy_tree "./$REPO_DIR/assets"    "./assets"

# Require destination dirs to exist after repo copy
[ -d "_includes" ] || { err "_includes/ does not exist after repo copy"; exit 1; }
[ -d "_layouts"  ] || { err "_layouts/ does not exist after repo copy"; exit 1; }

info "Step 3/3: Apply local override folders (if present)"

sync_dir_overwrite "__includes" "_includes"
sync_dir_overwrite "__layouts"  "_layouts"

info "All done."