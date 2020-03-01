#!/usr/bin/env bash

mv ../analyzing-gui-data/contents-v1.txt data/contents-v1.txt
mv ../analyzing-gui-data/selected-v1/* data/selected-v1/*

python3 extract_image_features.py
python3 preprocess.py