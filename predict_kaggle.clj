(ns predict_kaggle

  (:require [libpython-clj2.python :refer [py.- py.] :as py]
     [libpython-clj2.require :refer [require-python]]
     [scicloj.ml.dataset :as ds]
     [tech.v3.libs.arrow :as arrow]))

(require-python '[simpletransformers.classification
                  :as classification :reload])
(require-python '[pandas :as pd :reload])

(def model
  (classification/ClassificationModel "distilbert" "./outputs/checkpoint-667-epoch-1/"))


(def  test
  (->
   (ds/dataset "test.csv" {:key-fn keyword})
   (ds/select-columns [:text :id])))
   ;; (ds/head)



(def prediction
  (py/with-gil-stack-rc-context
    (->
     (py. model predict (-> test :text py/->py-list))
     py/->jvm
     first)))

(-> test
    (ds/add-column :target prediction)
    (ds/select-columns [:id :target])
    (ds/write-csv! "submission.csv"))
