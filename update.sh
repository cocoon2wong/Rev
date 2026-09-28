#!/bin/sh
#
# @Author: Conghao Wong
# @Date: 2025-03-24 17:16:55
# @LastEditors: Conghao Wong
# @LastEditTime: 2026-09-22 21:05:00
# @Github: https://cocoon2wong.github.io
# Copyright 2025 Conghao Wong, All Rights Reserved.
#

set -eu

echo "[INFO] Cleaning previous assembled working directories..."
# Since all io pages, assets, and configs are maintained strictly outside src/,
# ./src is completely transient and safe to wipe clean on each update.
rm -rf ./src
rm -rf ./public/assets
rm -rf ./.astro
rm -rf ./dist

echo "[INFO] Running preprocess..."
sh preprocess.sh

if [ ! -d "node_modules" ]; then
  echo "[INFO] node_modules not found. Installing dependencies..."
  npm install
fi

echo "[INFO] Starting Astro dev server..."
npm run dev
