#!/usr/bin/env bash
export _JAVA_OPTIONS="-Dlibpython_clj.manual_gil=true"
python -c 'from clojurebridge import cljbridge;cljbridge.load_clojure_file(clj_file="src/train.clj",mvn_local_repo="/home/carsten/.m2/repository")'
