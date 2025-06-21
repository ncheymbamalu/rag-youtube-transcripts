#!/usr/bin/env bash
make check && make ; make clean
dvc status --quiet || export CHANGES="./artifacts has been modified."
printenv CHANGES && make update_artifacts && unset CHANGES