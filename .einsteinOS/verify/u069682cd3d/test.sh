#!/bin/bash
set -e
cd "$(dirname "$0")/../../.."
[ -x build/test_unit4 ] || bash .einsteinOS/verify/u069682cd3d/build.sh
./build/test_unit4
