#!/bin/bash

REMOTE_USER="jonas"
REMOTE_HOST="77.235.243.6"
REMOTE_DIR="/volume1/BSP_data/Acoustics/SLU_Sailor1/Raw_data/2026"
LOCAL_DIR="data/raw"
FILE_LIST="data/tricky_bottom_files.txt"

mkdir -p "$LOCAL_DIR"

while IFS= read -r filename || [[ -n "$filename" ]]; do
    # Skip empty lines
    [[ -z "$filename" ]] && continue

    echo "Copying: $filename"

    scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${filename}" \
        "$LOCAL_DIR/"

    if [[ $? -ne 0 ]]; then
        echo "ERROR: Failed to copy $filename" >&2
    fi
done < "$FILE_LIST"