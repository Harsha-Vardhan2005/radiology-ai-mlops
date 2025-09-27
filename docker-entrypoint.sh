#!/bin/bash
set -e

echo "🚀 Starting Radiology AI MLOps Application..."

# Check if required environment variables are set
required_vars=("AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY" "S3_BUCKET_NAME")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Error: Required environment variable $var is not set"
        exit 1
    fi
done

echo "✅ Environment variables validated"

# Create necessary directories
mkdir -p /app/models /app/api/uploads /app/logs

# Set proper permissions (ignore errors for mounted volumes)
chmod 755 /app/models /app/api/uploads /app/logs 2>/dev/null || echo "⚠️  Warning: Could not set permissions on some directories (likely mounted volumes - this is normal)"

echo "📁 Directories created"

# Health check for AWS connectivity
echo "🔍 Testing AWS S3 connectivity..."
python -c "
import boto3
import os
try:
    s3 = boto3.client('s3', 
                     aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                     aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                     region_name=os.getenv('AWS_REGION', 'us-east-1'))
    s3.head_bucket(Bucket=os.getenv('S3_BUCKET_NAME'))
    print('✅ AWS S3 connection successful')
except Exception as e:
    print(f'❌ AWS S3 connection failed: {e}')
    exit(1)
"

# Pre-download model to avoid multiple downloads
echo "📥 Pre-downloading model..."
python -c "
from api.predict_service import download_model_from_s3
try:
    download_model_from_s3()
    print('✅ Model pre-downloaded successfully')
except Exception as e:
    print(f'❌ Model download failed: {e}')
    exit(1)
"

# Start the application
echo "🎯 Starting FastAPI application..."
if [ "$1" = "dev" ]; then
    echo "🔧 Running in development mode"
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
else
    echo "🚀 Running in production mode"
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
fi