(ns train
  (:require
   [scicloj.ml.dataset :as ds]
   [clojure.data.json :as json]
   [clj-yaml.core :as yaml]
   [tablecloth.api :as tc]
   [preprocess :refer [preprocess]]
   [simpletransformers]
   [scicloj.ml.core :as ml]
   [scicloj.ml.metamorph :as mm]
   [tech.v3.libs.arrow :as arrow]
   [confuse.binary-class-metrics :as confuse]
   [libpython-clj2.python.ffi :as ffi]
   [libpython-clj2.python :refer [py.- py.] :as py]))
   
;; (py/initialize!)
(println "-------- manual-gil: " ffi/manual-gil)
;; (println :lock-gil)
;; (def locked (ffi/lock-gil))


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
    :early_stopping_consider_epochs true

    :save_eval_checkpoints true
    :evaluate_during_training_steps 100
    :evaluate_during_training true
    :evaluate_during_training_silent false
    :evaluate_during_training_verbose false}
   params))




(def pd-train
   (->
    (arrow/stream->dataset "train.arrow" {:key-fn keyword})
    (tc/select-columns [:text :labels])))



(def pd-eval
  (->
   (arrow/stream->dataset "test.arrow" {:key-fn keyword})
   (tc/select-columns [:text :labels])))

(println :datasets-imported)

(try
  (py/with-manual-gil-stack-rc-context
    (let [pipe (ml/pipeline
                (mm/set-inference-target [:labels])
                (mm/model {:model-type :simpletransformers/classification
                           :model-args model-args
                           :eval_df pd-eval}))

          _ (println :run-fit)
          ctx-fit
          (ml/fit-pipe
           pd-train
           pipe)


          _ (println :run-transform)
          ctx-tf
          (ml/transform-pipe pd-eval pipe ctx-fit)


          actual (:labels pd-eval)
          predicted (-> ctx-tf :metamorph/data :labels)

          mcc (confuse/mcc actual predicted 1)]

      (spit "eval.json"

            (json/write-str
             {:train {:mcc mcc}}))))
  (println "training finished"))

  
