#!/usr/bin/env bash
# norax-sync.sh — sync changes between public and private Norax repos
# Usage:
#   bash norax-sync.sh public    — push public changes to clarboncy/norax
#   bash norax-sync.sh private   — push private changes to clarboncy/norax-c
#   bash norax-sync.sh pull-public  — pull public changes into private runtime
#   bash norax-sync.sh status    — show current state

set -euo pipefail
cd "$(dirname "$0")/.."
RUNTIME_REPO="$(pwd -P)"

require_repo_dir() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" || ! -d "$value/.git" ]]; then
    echo "$name must name an existing Git worktree." >&2
    exit 2
  fi
}

case "${1:-status}" in
  public)
    echo "=== Pushing to public repo (clarboncy/norax) ==="
    git checkout master
    git add -A
    if git diff --cached --quiet; then
      echo "No changes to push."
    else
      git commit -m "$(date +%Y-%m-%d) public update"
      git push public master
    fi
    echo "Done. Public repo updated."
    ;;

  private)
    echo "=== Pushing to private repo (clarboncy/norax-c) ==="
    git checkout private
    git add -A
    if git diff --cached --quiet; then
      echo "No changes to push."
    else
      git commit -m "$(date +%Y-%m-%d) private runtime update"
      git push private private
    fi
    git checkout master
    echo "Done. Private repo updated."
    ;;

  pull-public)
    echo "=== Pulling public changes into private runtime ==="
    git checkout master
    git pull public master
    echo ""
    echo "=== Merging into private branch ==="
    git checkout private
    git merge master -m "Merge public updates into private"
    git push private private
    git checkout master
    echo "Done. Private runtime updated with public changes."
    ;;

  sync-from-public)
    PUBLIC_REPO="${NORAX_PUBLIC_REPO_PATH:-}"
    PRIVATE_REPO="${NORAX_PRIVATE_REPO_PATH:-$RUNTIME_REPO}"
    require_repo_dir NORAX_PUBLIC_REPO_PATH "$PUBLIC_REPO"
    require_repo_dir NORAX_PRIVATE_REPO_PATH "$PRIVATE_REPO"
    echo "=== Pulling public changes from configured public worktree ==="
    cd "$PUBLIC_REPO"
    git pull origin master
    echo ""
    echo "=== Copying public files to private runtime ==="
    cd "$PRIVATE_REPO"
    git checkout master
    rsync -av --exclude='.git' --exclude='.env' --exclude='memory/' --exclude='.venv' \
      --exclude='training/checkpoints/' \
      --exclude='__pycache__' --exclude='*.pyc' \
      "$PUBLIC_REPO/" "$PRIVATE_REPO/"
    echo ""
    echo "Public files copied to private runtime."
    echo "Review with: git diff"
    echo "Commit with: bash norax-sync.sh public"
    ;;

  status)
    echo "=== Current branch ==="
    git branch --show-current
    echo ""
    echo "=== Branches ==="
    git branch -vv
    echo ""
    echo "=== Remotes ==="
    git remote -v
    echo ""
    echo "=== Uncommitted changes ==="
    git status --short | head -20
    if [ -z "$(git status --short)" ]; then
      echo "Clean working tree."
    fi
    ;;

  *)
    echo "Usage: bash norax-sync.sh {public|private|pull-public|sync-from-public|status}"
    echo ""
    echo "Commands:"
    echo "  public           Push public changes to clarboncy/norax"
    echo "  private          Push private changes to clarboncy/norax-c"
    echo "  pull-public      Pull public changes and merge into private branch"
    echo "  sync-from-public Pull from norax-public folder and copy to runtime"
    echo "  status           Show current state"
    ;;
esac
