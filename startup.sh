#!/bin/bash
# startup.sh - Launch Streamlit using the PORT Azure provides (or 8501 fallback)
set -e

# Azure provides PORT environment variable; fallback to 8501 for local test
PORT=${PORT:-8501}

# Make sure Streamlit serves on 0.0.0.0 so Azure can connect
export STREAMLIT_SERVER_HEADLESS=true
exec streamlit run Final.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false
