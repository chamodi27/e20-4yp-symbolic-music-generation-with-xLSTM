#!/bin/bash
# start.sh — Start the xLSTM Music Generation API server in a tmux session
#
# Behaviour:
#   - If the tmux session doesn't exist: creates it and starts the server
#   - If the session exists and server is running: just attaches to it
#   - If the session exists but server is NOT running: restarts the server
#
# Usage:
#   bash start.sh          — start or attach
#   bash start.sh --stop   — stop the server and kill the session

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================

# Path to the model checkpoint folder (required)
MODEL_PATH="/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/repos/helibrunna/output/xlstm_lmd_preprocessed_512d_4096ctx_12b/run_20260301-1906/checkpoint-46000-last"

# Conda environment name
ENV_NAME="xlstm-api"

# Server settings
API_HOST="0.0.0.0"
PORT=5059
CONTEXT_LENGTH=16384

# Tmux session name (fixed name avoids creating duplicates)
SESSION="xlstm-music-api"

# =============================================================================
# SCRIPT DIR — so the server is always run from the backend/ folder
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# STOP flag
# =============================================================================
if [ "$1" == "--stop" ]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Stopping session '$SESSION'..."
        tmux kill-session -t "$SESSION"
        echo "Done."
    else
        echo "No session '$SESSION' found. Nothing to stop."
    fi
    exit 0
fi

# =============================================================================
# Validate checkpoint path
# =============================================================================
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH"
    echo "Edit MODEL_PATH in start.sh to point to your checkpoint folder."
    exit 1
fi

# =============================================================================
# The command that runs the server
# =============================================================================
SERVER_CMD="MODEL_PATH='$MODEL_PATH' API_HOST='$API_HOST' PORT='$PORT' CONTEXT_LENGTH='$CONTEXT_LENGTH' conda run -n '$ENV_NAME' python '$SCRIPT_DIR/app.py'"

# =============================================================================
# MAIN LOGIC
# =============================================================================
if tmux has-session -t "$SESSION" 2>/dev/null; then
    # Session exists — check if app.py is still running inside it
    if tmux list-panes -t "$SESSION" -F "#{pane_current_command}" 2>/dev/null | grep -q "python\|conda"; then
        echo "Server is already running in tmux session '$SESSION'."
        echo "Attaching... (press Ctrl+B then D to detach)"
        echo ""
        tmux attach-session -t "$SESSION"
    else
        echo "Session '$SESSION' exists but server is not running. Restarting..."
        tmux send-keys -t "$SESSION" "$SERVER_CMD" Enter
        echo "Restarted. Attaching... (press Ctrl+B then D to detach)"
        echo ""
        tmux attach-session -t "$SESSION"
    fi
else
    # Session doesn't exist — create it and start the server
    echo "Creating tmux session '$SESSION' and starting server..."
    echo "  Model:   $MODEL_PATH"
    echo "  Server:  http://$API_HOST:$PORT"
    echo ""
    tmux new-session -d -s "$SESSION" -c "$SCRIPT_DIR"
    tmux send-keys -t "$SESSION" "$SERVER_CMD" Enter
    echo "Server starting in background. Attaching... (press Ctrl+B then D to detach)"
    echo ""
    sleep 1
    tmux attach-session -t "$SESSION"
fi
