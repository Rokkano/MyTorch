#!/bin/sh

DIR=src/
OUTPUT_FILE=cov/cov_empty.json

echo '{
  "gcovr/format_version": 0.11,
  "files": [' > "$OUTPUT_FILE"

FIRST=1
find "$DIR" -type f | while read -r FILE; do
  if [ $FIRST -eq 0 ]; then
    echo ',' >> "$OUTPUT_FILE"
  fi
  FIRST=0
  printf '    {\n      "file": "%s",\n      "functions": [],\n      "lines": []\n    }' "$FILE" >> "$OUTPUT_FILE"
done

echo '
  ]
}' >> "$OUTPUT_FILE"