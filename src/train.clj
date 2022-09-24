(ns train
  (:require [libpython-clj2.python :refer [py.- py.] :as py]
            [libpython-clj2.require :refer [require-python]]
            [scicloj.ml.dataset :as ds]
            [clojure.data.json :as json]
            [clj-yaml.core :as yaml]
            [tablecloth.api :as tc]
            [preprocess :refer [preprocess]]
            [tech.v3.libs.arrow :as arrow]))

(require-python '[simpletransformers.classification
                  :as classification])
(require-python '[pandas :as pd])

(def params
  (->
   (slurp "params.yaml")
   (yaml/parse-string)
   :train))

(def model-args
  (merge
   {"use_cuda" false
    "use_early_stopping" true
    "save_eval_checkpoints" true
    "evaluate_during_training_steps" 100
    "evaluate_during_training" true
    "evaluate_during_training_silent" true
    "evaluate_during_training_verbose" true}
   params))


(def pd-train
  (->
   (arrow/stream->dataset "train.arrow" {:key-fn keyword})
   (tc/select-columns [:text :labels])
   (tc/head 100)
   (tc/rows :as-seqs)
   (pd/DataFrame)))

(def pd-eval
  (->
   (arrow/stream->dataset "test.arrow" {:key-fn keyword})
   (tc/select-columns [:text :labels])
   (tc/head)
   (tc/rows :as-seqs)
   (pd/DataFrame)))

(def model
  (classification/ClassificationModel
   (:model_type model-args)
   (:model_name model-args)
   :args model-args))



(def _  (py. model train_model pd-train :eval_df pd-eval))

(println "training finished")
(shutdown-agents)


(comment

  (System/gc)
 (require '[libpython-clj2.python.gc])
 (libpython-clj2.python.gc/clear-reference-queue)
 (System/gc))
