#!/usr/bin/env bash
# Download CGMacros v1.0.0 from PhysioNet (open access, CC BY-NC-SA 4.0) into data/raw/cgmacros/.
# The raw files are git-ignored: the licence is non-commercial share-alike and the repo has no
# licence yet, so only aggregate results are committed.
# Cite the PhysioNet record (https://physionet.org/content/cgmacros/1.0.0/) and the data paper
# (https://pmc.ncbi.nlm.nih.gov/articles/PMC12462512/). Also mirrored on the AWS Open Data
# registry (https://registry.opendata.aws/cgmacros/).
# Not yet run end to end from here: the cloud session that wrote it could not reach PhysioNet,
# so check the folder layout after the first download.
set -euo pipefail
dest="$(dirname "$0")/../data/raw/cgmacros"
mkdir -p "$dest"
base="https://physionet.org/files/cgmacros/1.0.0"
wget -r -N -c -np -nH --cut-dirs=3 -R "*.jpg,index.html*" -P "$dest" "$base/"
# The participant files sit in CGMacros/CGMacros-0XX/CGMacros-0XX.csv; flatten them.
find "$dest" -name 'CGMacros-0*.csv' -exec mv -n {} "$dest/" \;
ls "$dest"/CGMacros-0*.csv | wc -l   # expect 45
