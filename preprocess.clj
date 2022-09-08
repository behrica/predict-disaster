(ns preprocess
   (:require [scicloj.ml.dataset :as ds]
             [tech.v3.libs.arrow :as arrow]))


(def  data
  (->
   (ds/dataset "train.csv" {:key-fn keyword})
   (ds/select-columns [:text :target])
   (ds/rename-columns {:target :labels})))

   

(def data-split
  (ds/train-test-split data))

(arrow/dataset->stream! (:train-ds data-split) "train.arrow")
(arrow/dataset->stream! (:test-ds data-split) "test.arrow")


(System/exit 0)



