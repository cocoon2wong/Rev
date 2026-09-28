#!/bin/sh
set -eu

REPO_URL="https://github.com/cocoon2wong/Project-Zero.git"
REPO_DIR="Project-Zero"
BRANCH="main"

info() { printf '%s\n' "[INFO] $*"; }
warn() { printf '%s\n' "[WARN] $*"; }
err()  { printf '%s\n' "[ERROR] $*"; exit 1; }

info "Step 1/4: Checking Project-Zero template repository..."

if [ -e "$REPO_DIR" ] || [ -L "$REPO_DIR" ]; then
  info "Project-Zero already exists ($REPO_DIR). Skipping clone/checkout."
else
  info "Cloning: $REPO_URL -> $REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
  (
    cd "$REPO_DIR"
    git checkout "$BRANCH" || true
  )
fi

[ -d "$REPO_DIR" ] || err "Project-Zero folder not found: $REPO_DIR"

info "Step 2/4: Copying framework components, layouts, styles, and utils from $REPO_DIR..."

mkdir -p ./src/components ./src/layouts ./src/styles ./src/utils ./public

for dir in components layouts styles utils; do
  if [ -d "$REPO_DIR/src/$dir" ]; then
    info "Syncing framework directory: src/$dir"
    mkdir -p "./src/$dir"
    cp -Rf "$REPO_DIR/src/$dir"/. "./src/$dir"/
  fi
done

if [ -d "$REPO_DIR/public" ]; then
  info "Syncing framework public assets: favicon, etc."
  cp -Rf "$REPO_DIR/public"/. "./public"/
fi

info "Step 3/4: Placing local assets and data..."

if [ -d "__assets" ]; then
  mkdir -p ./public/assets
  info "Applying local __assets -> public/assets"
  cp -Rf "__assets"/. "./public/assets"/
fi

if [ -d "__data" ]; then
  mkdir -p ./src/data
  info "Applying local __data -> src/data"
  cp -Rf "__data"/. "./src/data"/
fi

info "Step 4/4: Assembling pages from __pages/ into src/pages/..."

mkdir -p ./src/pages

if [ -d "__pages" ]; then
  info "Applying local __pages -> src/pages"
  cp -Rf "__pages"/. "./src/pages"/
fi

if [ -d "__posts" ]; then
  info "Placing blog posts: __posts/ -> src/pages/posts/"
  mkdir -p "./src/pages/posts"
  cp -Rf "__posts"/. "./src/pages/posts"/
fi

info "Preprocess completed successfully."