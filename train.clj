(ns preprocess
   (:require ;; [libpython-clj2.python :refer [py.- py.] :as py]
             ;; [libpython-clj2.require :refer [require-python]]
             [scicloj.ml.dataset :as ds]
             [tech.v3.libs.arrow :as arrow]))







(def  data
  (->
   (ds/dataset "train.csv" {:key-fn keyword})
   (ds/select-columns [:text :target])
   (ds/rename-columns {:target :labels})))
   ;; (ds/random 500)



   

(def data-split
  (ds/train-test-split data))

(arrow/dataset->stream! (:train-ds data-split) "train.arrow")
(arrow/dataset->stream! (:test-ds data-split) "test.arrow")






;; (def train-result
;;   (py/with-gil-stack-rc-context
;;     (let [
;;           train-df (pd/DataFrame (:train-ds data-split))
;;           eval-df  (pd/DataFrame (:test-ds data-split))
;;           model (classification/ClassificationModel

;;                  :use_cuda true
;;                  :model_type "bert"
;;                  :model_name "prajjwal1/bert-tiny"
;;                  :args
;;                  { ;; :silent true
;;                   :dataloader_num_workers 1
;;                   :process_count 1
;;                   :num_train_epochs 5
;;                   :logging_steps 1000
;;                   :evaluate_during_training true
;;                   :use_multiprocessing false})


;;           train-result (py. model train_model
;;                             train-df
;;                             :eval_df eval-df)]



;;       (py/->jvm train-result))))



;; (spit "train-result.edn"
;;       (with-out-str (clojure.data.json/pprint (second  train-result))))

;; (println "training finished")
