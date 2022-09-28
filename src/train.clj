(ns train
  (:require
   [scicloj.ml.dataset :as ds]
   [clojure.data.json :as json]
   [clj-yaml.core :as yaml]
   [tablecloth.api :as tc]
   [preprocess :refer [preprocess]]
   [scicloj.ml.core :as ml]
   [scicloj.ml.metamorph :as mm]
   [tech.v3.libs.arrow :as arrow]
   [simpletransformers]
   [confuse.binary-class-metrics :as confuse]
   [libpython-clj2.python.ffi :as ffi]
   [libpython-clj2.python :refer [py.- py.] :as py]))
   

   


(println "-------- manual-gil: " ffi/manual-gil)
(py/initialize!)


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
    :save_eval_checkpoints true
    :evaluate_during_training_steps 100
    :evaluate_during_training true
    :evaluate_during_training_silent false
    :evaluate_during_training_verbose false}
   params))

(println :gil-locked)
(def locked (ffi/lock-gil))


;; (py/initialize! :no-io-redirect? true)

(println :import-python-libs)

(def pd-train
   (->
    (arrow/stream->dataset "train.arrow" {:key-fn keyword})
    (tc/select-columns [:text :labels])))


(def pd-eval
  (->
   (arrow/stream->dataset "test.arrow" {:key-fn keyword})
   (tc/select-columns [:text :labels])))


(println :datasets-imported)

(def pipe (ml/pipeline
           (mm/set-inference-target [:labels])
           (mm/model {:model-type :simpletransformers/classification
                      :model-args model-args
                      :eval_df pd-eval})))


(def ctx-fit
  (ml/fit-pipe
   pd-train
   pipe))

   
(def ctx-tf
  (ml/transform-pipe pd-eval pipe ctx-fit))




(def actual (:labels pd-eval))
(def predicted (-> ctx-tf :metamorph/data :labels))

(def mcc (confuse/mcc actual predicted 1))



;; (def eval-result
;;   (->
;;    (py. model eval_model pd-eval)
;;    (py/->jvm)))

;; (println :evaluation-done)

(spit "eval.json"
      
     (json/write-str
      {:train {:mcc mcc}}))





(println "training finished")

(println :unlock-gil)
(ffi/unlock-gil locked)
