#!/usr/bin/env bash
#
# runner.sh — continuous translation loop for the epitaka translator.
#
#   ./runner.sh [lang] [model]
#
#   ./runner.sh si "gemini-3.7-flash"   # Sinhala, pinned model
#   ./runner.sh si                      # Sinhala, model fallback chain
#
# What it does:
#   1. Ensures the required SQLite data files exist in ./data (downloading
#      + unzipping them from the epitaka_app GitHub releases when missing).
#   2. Activates .venv if present.
#   3. Runs `python src/book_translator.py --lang <lang> --books preset ...`
#      in a loop: if all API keys are exhausted it sleeps 3h and retries
#      (translation resumes where it left off); on other errors it retries
#      after 60s; on success it exits.
#
set -u

SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 1. Required data files — download anything that's missing.
#    Override the data directory with:  DATA_DIR=/path/to/data ./runner.sh si
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-$(cd "$SCRIPT_DIR/data" 2>/dev/null && pwd || echo "$SCRIPT_DIR/data")}"
BASE_URL="${BASE_URL:-https://github.com/dhammanana/epitaka_app/releases/download/latest}"

# Each entry: "<db file expected in DATA_DIR>|<zip file name on the release>"
REQUIRED_FILES="
epitaka.db|epitaka.zip
dpd-dictionary.db|dpd-dictionary.zip
epitaka_en.db|epitaka_en.zip
epitaka_th.db|epitaka_th.zip
epitaka_si.db|epitaka_si.zip
epitaka_my_nissaya.db|epitaka_my_nissaya.zip
"

download_file() {
    _url="${1}"
    _dest="${2}"
    if command -v curl >/dev/null 2>&1; then
        curl -fSL --retry 3 -o "${_dest}" "${_url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${_dest}" "${_url}"
    else
        echo "[DATA] ERROR: neither curl nor wget is available — cannot download ${_url}" >&2
        return 1
    fi
}

ensure_data_files() {
    mkdir -p "$DATA_DIR"
    if ! command -v unzip >/dev/null 2>&1; then
        echo "[DATA] ERROR: 'unzip' is required but not installed." >&2
        return 1
    fi

    _missing=0
    echo "$REQUIRED_FILES" | while IFS='|' read -r _db _zip; do
        # skip blank lines
        _db="$(echo "${_db}" | tr -d '[:space:]')"
        _zip="$(echo "${_zip}" | tr -d '[:space:]')"
        [ -z "${_db}" ] && continue

        if [ -f "$DATA_DIR/${_db}" ]; then
            echo "[DATA] OK: ${_db} exists."
            continue
        fi

        echo "[DATA] Missing: ${_db} — downloading ${_zip} ..."
        _tmpzip="$(mktemp -t epitaka_data_XXXXXX.zip)"
        if download_file "$BASE_URL/${_zip}" "${_tmpzip}"; then
            unzip -o -q "${_tmpzip}" -d "$DATA_DIR"
            rm -f "${_tmpzip}"
            if [ -f "$DATA_DIR/${_db}" ]; then
                echo "[DATA] OK: ${_db} downloaded and extracted."
            else
                echo "[DATA] WARNING: ${_zip} extracted but ${_db} still not found." >&2
                echo "[DATA] Contents of $DATA_DIR:" >&2
                ls -la "$DATA_DIR" >&2
                # signal failure to the outer shell via a sentinel file
                touch "$DATA_DIR/.download_failed"
            fi
        else
            echo "[DATA] ERROR: failed to download $BASE_URL/${_zip}" >&2
            rm -f "${_tmpzip}"
            touch "$DATA_DIR/.download_failed"
        fi
    done

    if [ -f "$DATA_DIR/.download_failed" ]; then
        rm -f "$DATA_DIR/.download_failed"
        echo "[DATA] One or more data files could not be obtained. Fix the errors above and re-run." >&2
        return 1
    fi
    return 0
}

ensure_data_files || exit 1
export EPITAKA_DB="${EPITAKA_DB:-$DATA_DIR/epitaka.db}"

# ---------------------------------------------------------------------------
# 2. Python environment — create .venv and install requirements if needed
# ---------------------------------------------------------------------------
if [ ! -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    echo "[PYTHON] No .venv found — creating it (python3 -m venv .venv)..."
    if ! command -v python3 >/dev/null 2>&1; then
        echo "[PYTHON] ERROR: python3 is required but not installed." >&2
        exit 1
    fi
    python3 -m venv "$SCRIPT_DIR/.venv" || exit 1
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.venv/bin/activate"

VENV_PY="$SCRIPT_DIR/.venv/bin/python"
if ! "$VENV_PY" -c "import google.genai, dotenv, requests" 2>/dev/null; then
    echo "[PYTHON] Installing dependencies (pip install -r requirements.txt)..."
    "$VENV_PY" -m pip install -r "$SCRIPT_DIR/requirements.txt" || exit 1
else
    echo "[PYTHON] Dependencies OK."
fi

# ---------------------------------------------------------------------------
# 3. Language / model selection
# ---------------------------------------------------------------------------
LANG_CODE="${1:-}"
MODEL_NAME="${2:-}"

if [ -z "$LANG_CODE" ]; then
    read -rp "Enter target language code (e.g., si, th, en): " LANG_CODE
fi

if [ -z "$MODEL_NAME" ]; then
    read -rp "Enter model name (empty = automatic fallback chain, recommended): " MODEL_NAME
fi

# When no model is given we omit --model entirely so book_translator.py uses
# its FALLBACK_MODEL_CHAIN (gemini-3.7-flash -> ... -> gemini-3-flash-preview).
MODEL_ARGS=()
if [ -n "$MODEL_NAME" ]; then
    MODEL_ARGS=(--model "$MODEL_NAME")
fi

# 3 hours in seconds
SLEEP_DURATION=10800

echo "Starting continuous translation loop for lang='$LANG_CODE', model='${MODEL_NAME:-(fallback chain)}'..."
echo "Data dir: $DATA_DIR"

while true; do
    echo "=================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting translation run..."
    echo "=================================================="

    # Run command and capture output while streaming it to the terminal
    # Using tee to print output live AND capture it in a temporary log file
    TEMP_LOG=$(mktemp)
    python src/book_translator.py --lang "$LANG_CODE" --books preset "${MODEL_ARGS[@]}" 2>&1 | tee "$TEMP_LOG"

    EXIT_CODE=${PIPESTATUS[0]}

    # Check if the exhaustion line exists in the output
    if grep -q "All API keys exhausted" "$TEMP_LOG"; then
        echo ""
        echo "[WARNING] Detected API key exhaustion!"

        echo "[CLEANUP] Removing temp and log files..."
        # Using -f (force) to bypass zsh/bash interactive prompts when deleting >100 files
        rm -f /tmp/gem*
        rm -rf /tmp/book_translator_logs/*

        echo "[SLEEP] Waiting 3 hours before retrying (will resume at $(date -d '+3 hours' '+%H:%M:%S' 2>/dev/null || date -v+3H '+%H:%M:%S'))..."
        rm -f "$TEMP_LOG"
        sleep "$SLEEP_DURATION"
    else
        rm -f "$TEMP_LOG"
        if [ $EXIT_CODE -eq 0 ]; then
            echo "[SUCCESS] Translation finished successfully without key exhaustion."
            break
        else
            echo "[ERROR] Process exited with error code $EXIT_CODE (not API key exhaustion). Retrying in 60s..."
            sleep 60
        fi
    fi
done
