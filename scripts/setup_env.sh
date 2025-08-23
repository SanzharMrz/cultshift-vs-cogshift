#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "[setup] Repo root: $REPO_ROOT"

# Ensure Python available
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not found" >&2
  exit 1
fi

# Create venv if missing
if [ ! -d "$REPO_ROOT/venv" ]; then
  echo "[setup] Creating virtual environment at venv"
  python3 -m venv "$REPO_ROOT/venv"
fi

# Activate venv
echo "[setup] Activating virtual environment"
# shellcheck disable=SC1091
source "$REPO_ROOT/venv/bin/activate"

# Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel

# Install requirements
if [ -f "$REPO_ROOT/requirements.txt" ]; then
  echo "[setup] Installing requirements"
  pip install -r "$REPO_ROOT/requirements.txt"
fi

# Load environment from .env if present
if [ -f "$REPO_ROOT/.env" ]; then
  echo "[setup] Loading .env"
  set -a
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.env"
  set +a
else
  echo "[setup] .env not found at repo root (expected keys: GITHUB_TOKEN, H4_TOKEN, GITHUB_NAME, GITHUB_EMAIL)" >&2
fi

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "[warn] Required env var '$name' is not set" >&2
    return 1
  fi
}

# Configure Git identity from env
if require_env GITHUB_NAME && require_env GITHUB_EMAIL; then
  echo "[setup] Configuring global Git identity"
  git config --global user.name "$GITHUB_NAME"
  git config --global user.email "$GITHUB_EMAIL"
fi

# Initialize git repo if needed
if [ ! -d "$REPO_ROOT/.git" ]; then
  echo "[setup] Initializing git repository"
  git init
fi

# Authenticate GitHub using token if available
if [ -n "${GITHUB_TOKEN:-}" ]; then
  if command -v gh >/dev/null 2>&1; then
    echo "[setup] Authenticating GitHub via gh"
    # Use non-interactive login with provided token
    echo "$GITHUB_TOKEN" | gh auth login --hostname github.com --with-token
    gh auth setup-git
  else
    echo "[setup] gh not found; configuring Git credential store for GitHub"
    git config --global credential.helper store
    CRED_FILE="$HOME/.git-credentials"
    GITHUB_CRED="https://x-access-token:$GITHUB_TOKEN@github.com"
    if [ -f "$CRED_FILE" ] && grep -q "github.com" "$CRED_FILE"; then
      :
    else
      printf "%s\n" "$GITHUB_CRED" >> "$CRED_FILE"
    fi
  fi
else
  echo "[warn] GITHUB_TOKEN not set; skipping GitHub auth"
fi

# Hugging Face login using H4_TOKEN (mirrored to standard vars)
if [ -n "${H4_TOKEN:-}" ]; then
  export HF_TOKEN="${HF_TOKEN:-$H4_TOKEN}"
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$H4_TOKEN}"
  if command -v huggingface-cli >/dev/null 2>&1; then
    echo "[setup] Logging into Hugging Face CLI"
    huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN" --add-to-git-credential || true
  else
    echo "[warn] huggingface-cli not found; token exported only"
  fi
else
  echo "[warn] H4_TOKEN not set; skipping Hugging Face login"
fi

echo "[setup] Done"


