(ns train
  (:require
   [scicloj.ml.dataset :as ds]
   [clojure.data.json :as json]
   [clj-yaml.core :as yaml]
   [tablecloth.api :as tc]
   [preprocess :refer [preprocess]]

   [tech.v3.libs.arrow :as arrow]
   [libpython-clj2.python.ffi :as ffi]
   [libpython-clj2.python :refer [py.- py.] :as py]))
   

   


(println "-------- manual-gil: " ffi/manual-gil)
;(py/initialize!)




(def params
  (->
   (slurp "params.yaml")
   (yaml/parse-string)
   :train))

(def model-args
  (merge

   {:use_multiprocessing false
    :use_multiprocessing_for_evaluation false
    :process_count 1

    :use_cuda true
    :use_early_stopping true
    :save_eval_checkpoints false
    ;:evaluate_during_training_steps 100
    :evaluate_during_training true
    :evaluate_during_training_silent true
    ;:evaluate_during_training_verbose true
  }
   params))

(def locked (ffi/lock-gil))
(println :gil-locked)


(println :import-python-libs)
(require
  '[libpython-clj2.require :as py-req])
(py-req/require-python '[pandas :as pd])
(py-req/require-python '[simpletransformers.classification :as st])

(println :python-libs-imported)

(def pd-train

   (->
    (arrow/stream->dataset "train.arrow" {:key-fn keyword})
    (tc/select-columns [:text :labels])
    ;(tc/head 102)
    (tc/rows :as-seqs)
    (pd/DataFrame)))

(def pd-eval

  (->
   (arrow/stream->dataset "test.arrow" {:key-fn keyword})
   (tc/select-columns [:text :labels])
   ;(tc/head 157)
   (tc/rows :as-seqs)
   (pd/DataFrame)))


(println :datasets-imported)

(def model (st/ClassificationModel
            ;; "bert" "prajjwal1/bert-tiny"
            (:model_type model-args)
            (:model_name model-args)
            :use_cuda (:use_cuda model-args)
            :args model-args))


(println :model-created)
(def train-result  (py. model train_model pd-train :eval_df pd-eval))


(println :model-trained)

(def eval-result
  (->
   (py. model eval_model pd-eval)
   (py/->jvm)))

(println :evaluation-done)

(spit "eval.json"
      (json/write-str
       {:train

        (-> eval-result first (select-keys ["mcc"]))}))


(println "training finished")

(println :unlock-gil)
(ffi/unlock-gil locked)
