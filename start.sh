#!/usr/bin/env bash

python manage.py migrate --noinput
python manage.py collectstatic --noinput

gunicorn AI_Career_Recommender.wsgi:application --bind 0.0.0.0:$PORT --workers 1
