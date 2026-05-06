#!/usr/bin/env bash
# pull-whisper.sh - Pull latest changes from ../whisper.cpp/examples and ../whisper.cpp/include
# Only copies files that already exist in the local version
set -euo pipefail

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHISPER_CPP="${SCRIPT_DIR}/../whisper.cpp"

if [ ! -d "$WHISPER_CPP" ]; then
    echo "ERROR: whisper.cpp directory not found at $WHISPER_CPP"
    exit 1
fi

echo "=== Pulling from whisper.cpp (only locally existing files) ==="
echo "Source: $WHISPER_CPP"
echo "Target: $SCRIPT_DIR"

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - no files will be copied"
fi
echo ""

# Function to check if a file exists locally and copy it
pull_file() {
    local src_file="$1"
    local rel_path="${src_file#$WHISPER_CPP/}"
    local dest_file="$SCRIPT_DIR/$rel_path"
    local dest_dir="$(dirname "$dest_file")"

    # Check if file exists locally
    if [ ! -f "$dest_file" ]; then
        return
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] Would copy: $src_file"
        echo "           To:      $dest_file"
        return
    fi

    mkdir -p "$dest_dir"
    cp -f "$src_file" "$dest_file"
    echo "  -> $dest_file"
}

# Map whisper.cpp paths to local paths
map_path() {
    local src_file="$1"
    local rel_path="${src_file#$WHISPER_CPP/}"
    local basename="$(basename "$rel_path")"
    local dir="$(dirname "$rel_path")"

    # Determine target directory based on source location
    case "$src_file" in
        "$WHISPER_CPP/examples"/*)
            case "$basename" in
                *.cpp)
                    echo "src/$basename"
                    ;;
                *.h)
                    echo "include/$basename"
                    ;;
                *)
                    # For other extensions, preserve directory structure
                    echo "$rel_path"
                    ;;
            esac
            ;;
        "$WHISPER_CPP/include"/*)
            echo "include/$basename"
            ;;
        *)
            echo "$rel_path"
            ;;
    esac
}

# Pull examples files that exist locally
echo "Pulling examples/ files that exist locally..."
find "$WHISPER_CPP/examples" -type f | while read -r file; do
    rel_path=$(map_path "$file")
    dest_file="$SCRIPT_DIR/$rel_path"
    if [ -f "$dest_file" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY-RUN] Would copy: $file"
            echo "           To:      $dest_file"
        else
            dest_dir="$(dirname "$dest_file")"
            mkdir -p "$dest_dir"
            cp -f "$file" "$dest_file"
            echo "  -> $dest_file"
        fi
    fi
done

# Pull include files that exist locally
echo ""
echo "Pulling include/ files that exist locally..."
find "$WHISPER_CPP/include" -type f | while read -r file; do
    rel_path=$(map_path "$file")
    dest_file="$SCRIPT_DIR/$rel_path"
    if [ -f "$dest_file" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY-RUN] Would copy: $file"
            echo "           To:      $dest_file"
        else
            dest_dir="$(dirname "$dest_file")"
            mkdir -p "$dest_dir"
            cp -f "$file" "$dest_file"
            echo "  -> $dest_file"
        fi
    fi
done

echo ""
echo "=== Done ==="
if [ "$DRY_RUN" = true ]; then
    echo "Dry run complete. No files were modified."
else
    echo "Files updated. Review changes with: git diff"
fi
