#!/bin/sh
set -e

echo "Checking ML model artifact..."
if [ ! -f "/app/dengues/ml_model/dchss_model_pure_python.pkl" ]; then
    echo "⚠️  WARNING: dchss_model_pure_python.pkl not found in dengues/ml_model/"
    echo "⚠️  Prediction endpoint will return 503 until the model file is placed there."
fi

echo "Running migrations..."
python manage.py makemigrations --noinput
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput --clear || true

echo "Starting Gunicorn server..."
exec gunicorn dengue.wsgi:application --bind 0.0.0.0:8000 --workers 3