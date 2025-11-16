#!/bin/bash

# Movie ROI Predictor - Streamlit App Runner
# This script sets up and runs the Streamlit application

echo "🎬 Starting Movie ROI Predictor..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check database connection
echo "🗄️ Checking database connection..."
python -c "
from utils.database import test_database_connection
import sys
if not test_database_connection():
    print('❌ Database connection failed!')
    print('Please ensure PostgreSQL is running and accessible.')
    sys.exit(1)
else:
    print('✅ Database connection successful!')
"

# Run Streamlit app
echo "🚀 Starting Streamlit application..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0


