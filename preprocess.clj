(ns preprocess
   (:require
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
      (tc/rename-columns {:target :labels})))



(def  data
  (->
   (tc/dataset "train.csv" {:key-fn keyword})
   preprocess))



(def data-split
  (first (tc/split->seq data :holdout {:seed 12345})))

(arrow/dataset->stream! (:train data-split) "train.arrow")
(arrow/dataset->stream! (:test data-split) "test.arrow")

(shutdown-agents)
;; (System/exit 0)
