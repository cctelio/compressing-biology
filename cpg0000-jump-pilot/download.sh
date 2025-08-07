#!/bin/bash

REMOTE="jump"
REMOTE_PATH="cellpainting-gallery/cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1"
LOCAL_DIR="/folder1/folder2/cpg0000-jump-pilot"
RCLONE_CONFIG="rclone.conf"

PLATES=(BR00116991 BR00116992 BR00116993 BR00116994 BR00116995 BR00117010 BR00117011 BR00117012 BR00117013 BR00117015 BR00117016 BR00117017 BR00117019 BR00117024 BR00117025 BR00117026)

mkdir -p "$LOCAL_DIR/illum"
mkdir -p "$LOCAL_DIR/images"

# Download illum files per plate
for PLATE in "${PLATES[@]}"; do
    echo "Downloading illum files for plate: $PLATE"
    rclone copy "$REMOTE:$REMOTE_PATH/illum/$PLATE" "$LOCAL_DIR/illum/$PLATE" \
        --config "$RCLONE_CONFIG" --create-empty-src-dirs --no-check-dest --disable-http2 --transfers 128 --multi-thread-streams 128 -v
done

# List all image files once
echo "Listing all image files..."
ALL_FILES=$(rclone ls "$REMOTE:$REMOTE_PATH/images" --config "$RCLONE_CONFIG" | awk '{print $2}')

# Download image files per plate using rclone include
for PLATE in "${PLATES[@]}"; do
    echo "Downloading image files for plate: $PLATE"
    IMAGE_DIR="$LOCAL_DIR/images/$PLATE"
    mkdir -p "$IMAGE_DIR"

    # Create a temporary include file
    INCLUDE_FILE=$(mktemp)
    echo "$ALL_FILES" | grep "^$PLATE" > "$INCLUDE_FILE"

    rclone copy "$REMOTE:$REMOTE_PATH/images" "$IMAGE_DIR" \
        --config "$RCLONE_CONFIG" --files-from "$INCLUDE_FILE" --no-check-dest --disable-http2 --transfers 128 --multi-thread-streams 128 -v

    rm "$INCLUDE_FILE"
done