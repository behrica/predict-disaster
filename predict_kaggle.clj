(ns predict_kaggle

  (:require [libpython-clj2.python :refer [py.- py.] :as py]
            [libpython-clj2.require :refer [require-python]]
            [scicloj.ml.dataset :as ds]
            [clojure.data.json :as json]
            [clj-yaml.core :as yaml]
            [tech.v3.libs.arrow :as arrow]))

(require-python '[simpletransformers.classification
                  :as classification :reload])
(require-python '[pandas :as pd :reload])

(def params
  (->
   (slurp "params.yaml")
   (yaml/parse-string)
   :train))

(def model
  (classification/ClassificationModel
   (:model_type params)
   "./outputs"))


(def  test
  (->
   (ds/dataset "test.csv" {:key-fn keyword})
   (ds/select-columns [:text :id])))



(def prediction
  (py/with-gil-stack-rc-context
    (->
     (py. model predict (-> test :text py/->py-list))
     py/->jvm
     first)))

(-> test
    (ds/add-column :target prediction)
    (ds/select-columns [:id :target])
    (ds/write-csv! "kaggle_submission.csv"))
