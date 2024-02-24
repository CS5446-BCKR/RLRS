#!/usr/bin/env bash

set -Eeuo pipefail

wget -P data/ https://files.grouplens.org/datasets/movielens/ml-1m.zip --no-check-certificate
pushd data
unzip ml-1m.zip && rm ml-1m.zip
popd
