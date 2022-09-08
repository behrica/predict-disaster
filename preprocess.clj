(ns preprocess
   (:require [scicloj.ml.dataset :as ds]
             [tablecloth.api :as tc]
             [tech.v3.libs.arrow :as arrow]))


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



(def  data
  (->
   (ds/dataset "train.csv" {:key-fn keyword})
   preprocess))


(def data-split
  (ds/train-test-split data))

(arrow/dataset->stream! (:train-ds data-split) "train.arrow")
(arrow/dataset->stream! (:test-ds data-split) "test.arrow")

(shutdown-agents)
;; (System/exit 0)
