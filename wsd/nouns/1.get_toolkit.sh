#!/bin/bash

set -e

GIT_URL=git@github.com:getalp/disambiguate.git

TARGET_DIRECTORY="$(dirname "$0")"/disambiguate

rm -rf $TARGET_DIRECTORY

git clone $GIT_URL $TARGET_DIRECTORY

