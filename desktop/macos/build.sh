#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
OUTPUT="${1:-$ROOT/dist}"
APP="$OUTPUT/Orbit.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
swiftc -parse-as-library "$ROOT/desktop/macos/OrbitApp.swift" -o "$OUTPUT/Orbit-arm64" -framework AppKit -framework WebKit -target arm64-apple-macos13.0
swiftc -parse-as-library "$ROOT/desktop/macos/OrbitApp.swift" -o "$OUTPUT/Orbit-x86_64" -framework AppKit -framework WebKit -target x86_64-apple-macos13.0
lipo -create "$OUTPUT/Orbit-arm64" "$OUTPUT/Orbit-x86_64" -output "$APP/Contents/MacOS/Orbit"
rm -f "$OUTPUT/Orbit-arm64" "$OUTPUT/Orbit-x86_64"
cp "$ROOT/desktop/macos/Info.plist" "$APP/Contents/Info.plist"
cp "$ROOT/install.sh" "$APP/Contents/Resources/install.sh"
cp "$ROOT/orbit/static/orbit-logo.png" "$APP/Contents/Resources/orbit-logo.png"
cp "$ROOT/orbit/static/orbit-logo-transparent.png" "$APP/Contents/Resources/orbit-logo-transparent.png"
ICONSET="$OUTPUT/Orbit.iconset"
rm -rf "$ICONSET"
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
  sips -z "$size" "$size" "$ROOT/orbit/static/orbit-logo.png" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z "$double" "$double" "$ROOT/orbit/static/orbit-logo.png" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/Orbit.icns"
rm -rf "$ICONSET"
codesign --force --deep --sign "${ORBIT_CODESIGN_IDENTITY:--}" "$APP"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$OUTPUT/Orbit-macOS-universal.zip"
printf '%s\n' "$OUTPUT/Orbit-macOS-universal.zip"
