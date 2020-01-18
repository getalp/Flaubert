#!/bin/bash

set -e

DISAMBIGUATE_GIT_URL=git@github.com:getalp/disambiguate.git
UFSAC_GIT_URL=git@github.com:getalp/UFSAC.git

DISAMBIGUATE_TARGET_DIRECTORY="$(dirname "$0")"/disambiguate
UFSAC_TARGET_DIRECTORY="$(dirname "$0")"/disambiguate/UFSAC

rm -rf $DISAMBIGUATE_TARGET_DIRECTORY
rm -rf $UFSAC_TARGET_DIRECTORY

git clone $DISAMBIGUATE_GIT_URL $DISAMBIGUATE_TARGET_DIRECTORY
git clone $UFSAC_GIT_URL $UFSAC_TARGET_DIRECTORY

$UFSAC_TARGET_DIRECTORY/java/install.sh
$DISAMBIGUATE_TARGET_DIRECTORY/java/compile.sh
