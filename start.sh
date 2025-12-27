#!/usr/bin/env bash

python manage.py migrate --noinput
python manage.py collectstatic --noinput

gunicorn ai_recommender.wsgi:application --bind 0.0.0.0:$PORT --workers 1
