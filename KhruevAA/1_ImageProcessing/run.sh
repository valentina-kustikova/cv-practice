#!/bin/bash


INPUT_IMAGE=${1:-"images/test.jpg"}

if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Error: Image '$INPUT_IMAGE' not found!"
    echo "Usage: ./run_all_filters.sh <path_to_image>"
    exit 1
fi

OUTPUT_DIR="images/results"
mkdir -p "$OUTPUT_DIR"

FILENAME=$(basename -- "$INPUT_IMAGE")
EXTENSION="${FILENAME##*.}"
NAME="${FILENAME%.*}"

echo "=================================================="
echo "Input image    : $INPUT_IMAGE"
echo "Results folder : $OUTPUT_DIR"
echo "=================================================="

FILTERS=("grayscale" "antique" "infrared" "matte" "noise" "neon")

echo "Applying filter: resize (1280x720)..."
python run.py -i "$INPUT_IMAGE" -o "$OUTPUT_DIR/${NAME}_resize.${EXTENSION}" -f resize --width 1280 --height 720

echo "Applying filter: fade (factor 0.7)..."
python run.py -i "$INPUT_IMAGE" -o "$OUTPUT_DIR/${NAME}_fade.${EXTENSION}" -f fade --factor 0.7

for FILTER in "${FILTERS[@]}"; do
    echo "Applying filter: $FILTER..."
    python run.py -i "$INPUT_IMAGE" -o "$OUTPUT_DIR/${NAME}_${FILTER}.${EXTENSION}" -f "$FILTER"
done

echo "=================================================="
echo "All filters applied successfully!"
echo "You can check the results in the directory: $OUTPUT_DIR"