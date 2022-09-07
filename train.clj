

(ns train
   (:require [libpython-clj2.python :refer [py.- py.] :as py]
             [libpython-clj2.require :refer [require-python]]
             [scicloj.ml.dataset :as ds]))



(require-python '[simpletransformers.classification
                  :as classification :reload])
(require-python '[pandas :as pd :reload])






(def  data
  (->
   (ds/dataset "train.csv" {:key-fn keyword})
   (ds/select-columns [:text :target])
   (ds/rename-columns {:target :labels})
   (ds/random 100)))

(def data-split
  (ds/train-test-split data))





(def train-result
  (py/with-gil-stack-rc-context
    (let [
          train-df (pd/DataFrame (:train-ds data-split))
          eval-df  (pd/DataFrame (:test-ds data-split))
          model (classification/ClassificationModel

                 :use_cuda false
                 :model_type "bert"
                 :model_name "prajjwal1/bert-tiny"
                 :args
                 { ;; :silent true
                  :dataloader_num_workers 1
                  :process_count 2
                  :num_train_epochs 1
                  :evaluate_during_training true
                  :evaluate_during_training_verbose true
                  :use_multiprocessing false
                  :overwrite_output_dir true})

          train-result (py. model train_model
                            train-df
                            :eval_df eval-df)]



      (py/->jvm train-result))))



(spit "train-result.edn"
      (with-out-str (clojure.data.json/pprint (second  train-result))))
