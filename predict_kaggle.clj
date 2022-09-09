(ns predict_kaggle

  (:require [libpython-clj2.python :refer [py.- py.] :as py]
            [libpython-clj2.require :refer [require-python]]
            [scicloj.ml.dataset :as ds]
            [clojure.data.json :as json]
            [clj-yaml.core :as yaml]
            [tablecloth.api :as tc]

            [tech.v3.libs.arrow :as arrow]))

(require-python '[simpletransformers.classification
                  :as classification :reload])
(require-python '[pandas :as pd :reload])

(defn preprocess [ds]
  (-> ds
      (tc/replace-missing [:location] :value "unknown")
      (tc/replace-missing [:keyword] :value "")
      (tc/add-or-replace-column
       :text
       (fn [ds] (map (fn [text location keyword]
                      (format "Keyword: %s\nLocation: %s.\n%s"
                              keyword location text))
                    (:text ds)
                    (:location ds)
                    (:keyword ds))))
      (ds/rename-columns {:target :labels})))


(def params
  (->
   (slurp "params.yaml")
   (yaml/parse-string)
   :train))

(def model
  (classification/ClassificationModel
   (:model_type params)
   "./outputs/best_model/"))


(def  test
  (->
   (ds/dataset "test.csv" {:key-fn keyword})
   preprocess))



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
