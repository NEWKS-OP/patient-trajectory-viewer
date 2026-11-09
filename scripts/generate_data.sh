#!/usr/bin/env bash
# Download Synthea and generate the synthetic population used by the viewer.
# Requires Java 17+. Takes about 2 minutes.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f synthea.jar ]; then
  curl -L -o synthea.jar \
    https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar
fi

# 600 living adults aged 30-85 in Massachusetts (deceased patients are kept too),
# fixed seeds and reference date so the output is reproducible.
java -jar synthea.jar -c synthea.properties -s 42 -cs 42 -r 20260101 -p 600 -a 30-85 Massachusetts

mkdir -p ../data/raw
for f in patients encounters conditions medications observations; do
  cp "output/csv/$f.csv" ../data/raw/
done
echo "CSV files copied to data/raw/. Now run: python -m pipeline.build"
