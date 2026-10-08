#!/usr/bin/env bash
# Installs everything the benchmark needs except system packages:
#   bench/.venv/            Python venv: numpy, pyarrow, scikit-learn (requirements.txt)
#   bench/.tools/weka.jar   Weka 3.8.7 (J48) and bounce.jar (a Weka dependency)
#   bench/.tools/classes/   the compiled J48 adapter
#   bench/.tools/yadt/      YaDT 2.3.0 Linux binary (dTcmd + libtbb)
# and builds ./tree_cpu. System packages: python3, R (rpart ships with it), a
# JDK 11+, taskset (util-linux), unzip, curl.
#
# YaDT is free for research and educational use only and may not be
# redistributed (bench/.tools/yadt/LICENCE.txt): every user downloads their own
# copy, so it is fetched here and never kept in the repository.

set -euo pipefail
cd "$(dirname "$0")"
TOOLS=.tools
mkdir -p "$TOOLS"

missing=()
for command in python3 Rscript java javac taskset unzip curl make g++; do
  command -v "$command" >/dev/null || missing+=("$command")
done
if ((${#missing[@]})); then
  echo "missing: ${missing[*]}" >&2
  exit 1
fi
Rscript -e 'suppressMessages(library(rpart))' || { echo "R package rpart missing" >&2; exit 1; }

make -C .. cpu

[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt

maven=https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka
[ -f "$TOOLS/weka.jar" ] ||
  curl -sSfL -o "$TOOLS/weka.jar" "$maven/weka-stable/3.8.7/weka-stable-3.8.7.jar"
[ -f "$TOOLS/bounce.jar" ] ||
  curl -sSfL -o "$TOOLS/bounce.jar" "$maven/thirdparty/bounce/0.18/bounce-0.18.jar"
mkdir -p "$TOOLS/classes"
javac -nowarn -d "$TOOLS/classes" -cp "$TOOLS/weka.jar:$TOOLS/bounce.jar" adapters/J48Fit.java

if [ ! -x "$TOOLS/yadt/dTcmd" ]; then
  curl -sSfL -o "$TOOLS/yadt.zip" https://pages.di.unipi.it/ruggieri/YaDT/YaDT2.3.0.zip
  unzip -oq "$TOOLS/yadt.zip" -d "$TOOLS"
  rm "$TOOLS/yadt.zip"
  chmod +x "$TOOLS/yadt/dTcmd"
fi

echo "ready:"
echo "  $(.venv/bin/python -c 'import sklearn; print("scikit-learn", sklearn.__version__)')"
echo "  R $(Rscript -e 'cat(as.character(getRversion()))'), rpart $(Rscript -e 'cat(as.character(packageVersion("rpart")))')"
echo "  $(java -version 2>&1 | head -1), Weka 3.8.7"
echo "  $(LD_LIBRARY_PATH=$TOOLS/yadt $TOOLS/yadt/dTcmd 2>&1 | head -1)"
command -v /usr/bin/time >/dev/null || echo "  (optional) GNU time not installed: needed only for manual memory cross-checks"
