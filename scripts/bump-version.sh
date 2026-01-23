#!/usr/bin/env bash
# Bump version in all locations that need manual updates.
# Usage: ./scripts/bump-version.sh 0.8.0

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <new-version>"
    echo "Example: $0 0.8.0"
    exit 1
fi

NEW_VERSION="$1"

# Validate version format
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format X.Y.Z (e.g., 0.8.0)"
    exit 1
fi

echo "Bumping version to $NEW_VERSION"

# pyproject.toml (source of truth)
sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
echo "  Updated pyproject.toml"

# server.json (MCP registry manifest)
sed -i '' "s/\"version\": \"[0-9]*\.[0-9]*\.[0-9]*\"/\"version\": \"$NEW_VERSION\"/g" server.json
echo "  Updated server.json"

# Verify
echo ""
echo "Verification:"
grep -n "version" pyproject.toml | head -1
grep -n "version" server.json | head -2

echo ""
echo "Done! Don't forget to:"
echo "  1. Update CHANGELOG.md"
echo "  2. Commit: git commit -am 'chore: bump version to $NEW_VERSION'"
echo "  3. Tag: git tag v$NEW_VERSION"
echo "  4. Push: git push && git push --tags"
