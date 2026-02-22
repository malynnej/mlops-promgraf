#!/usr/bin/env bash
# Simple traffic generator for /predict

API_URL=${API_URL:-http://localhost:8080/predict}
REQUESTS=${REQUESTS:-100}
SLEEP_SECONDS=${SLEEP_SECONDS:-0.5}

echo "Sending $REQUESTS requests to $API_URL ..."

for i in $(seq 1 "$REQUESTS"); do
  curl -s -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    --data "{
      \"temp\": 0.24,
      \"atemp\": 0.2879,
      \"hum\": 0.81,
      \"windspeed\": 0.0,
      \"mnth\": 1,
      \"hr\": $(( (i - 1) % 24 )),
      \"weekday\": $(( (i - 1) % 7 )),
      \"season\": 1,
      \"holiday\": 0,
      \"workingday\": 1,
      \"weathersit\": 1,
      \"dteday\": \"2011-01-01\"
    }" >/dev/null

  if (( i % 10 == 0 )); then
    echo "Sent $i requests..."
  fi

  sleep "$SLEEP_SECONDS"
done

echo "Done."
