

(ns train
   (:require [libpython-clj2.python :refer [py.- py.] :as py]
             [libpython-clj2.require :refer [require-python]]))



(require-python '[simpletransformers.classification
                  :as classification :reload])
(require-python '[pandas :as pd :reload])



(def  train-data  [
                   ["Example sentence belonging to class 1" 1]
                   ["Example sentence belonging to class 0" 0]])



(def eval-data  [
                 ["Example eval sentence belonging to class 1" 1]
                 ["Example eval sentence belonging to class 0" 0]])

(py/with-gil-stack-rc-context
  (let [
        train-df (pd/DataFrame train-data)
        eval-df (pd/DataFrame eval-data)
        model (classification/ClassificationModel

               :use_cuda false
               :model_type "bert"
               :model_name "prajjwal1/bert-tiny"
               :args
               (py/->py-dict

                {;; :silent true
                 :dataloader_num_workers 1
                 :process_count 2
                 :num_train_epochs 10
                 :use_multiprocessing false
                 :overwrite_output_dir true}))

        x (py. model train_model train-df)]



    (println :x x)
    (flush)))
