#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=false
python -c 'from clojurebridge import cljbridge;cljbridge.load_clojure_file(clj_file="train.clj",mvn_local_repo="/home/carsten/.m2/repository")'
