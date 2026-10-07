#!/usr/bin/env bash
# Downloads VisA (Amazon, CC BY 4.0) into data/. About 1.8 GB.
# Check the license and URL at https://github.com/amazon-science/spot-diff before running.
set -euo pipefail
mkdir -p data
cd data
curl -L -o VisA_20220922.tar https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
mkdir -p VisA
tar -xf VisA_20220922.tar -C VisA
echo "VisA extracted to data/VisA (expects data/VisA/split_csv/1cls.csv)"
