(ns train
  (:require
   [scicloj.ml.dataset :as ds]
   [clojure.data.json :as json]
   [clj-yaml.core :as yaml]
   [tablecloth.api :as tc]
   [preprocess :refer [preprocess]]
   [tech.v3.libs.arrow :as arrow]))
   


(def params
  (->
   (slurp "params.yaml")
   (yaml/parse-string)
   :train))

(def model-args
  (merge

   {;; "use_multiprocessing" false
    ;; "process_count" 1
    "use_cuda" true
    "use_early_stopping" true
    "save_eval_checkpoints" true
    ;; "evaluate_during_training_steps" 50
    "evaluate_during_training" true
    "evaluate_during_training_silent" false
    "evaluate_during_training_verbose" true}
   params))


(require
         '[libpython-clj2.python :refer [py.- py.] :as py])

(py/initialize! :no-io-redirect? true)

(def pd (py/import-module "pandas"))
(def st (py/import-module "simpletransformers.classification"))

(def pd-train
  ((py/py.- pd DataFrame)
   (->
    (arrow/stream->dataset "train.arrow" {:key-fn keyword})
    (tc/select-columns [:text :labels])
    ;; (tc/head 102)
    (tc/rows :as-seqs))))

(def pd-eval
  ((py/py.- pd DataFrame)
   (->
    (arrow/stream->dataset "test.arrow" {:key-fn keyword})
    (tc/select-columns [:text :labels])
    ;; (tc/head 157)
    (tc/rows :as-seqs))))

(def model ((py.- st ClassificationModel)
            "bert" "prajjwal1/bert-tiny"

             ;; (:model_type model-args)
             ;; (:model_name model-args)
            :use_cuda true
            :args model-args))

(def train-result  (py. model train_model pd-train :eval_df pd-eval))

(def eval-result
  (->
   (py. model eval_model pd-eval)
   (py/->jvm)))

(spit "eval.json"
 (json/write-str
  {:train

   (-> eval-result first (select-keys ["mcc"]))}))

(println "training finished")
(shutdown-agents)


