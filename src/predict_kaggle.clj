(ns predict_kaggle

  (:require [libpython-clj2.python :refer [py.- py.] :as py]
            [libpython-clj2.require :refer [require-python]]
            [scicloj.ml.dataset :as ds]
            [clojure.data.json :as json]
            [clj-yaml.core :as yaml]
            [tablecloth.api :as tc]
            [preprocess :refer [preprocess]]
            [tech.v3.libs.arrow :as arrow]))

(require-python '[simpletransformers.classification
                  :as classification])
;; (require-python '[pandas :as pd])
;;
;;


(defn predict-kaggle [_]
  (let [params
        (->
         (slurp "params.yaml")
         (yaml/parse-string)
         :train)

        model
        (classification/ClassificationModel
         (:model_type params)
         "./outputs/best_model/")

        test
        (->
         (ds/dataset "test.csv" {:key-fn keyword})
         preprocess)

        prediction
        (py/with-gil-stack-rc-context
          (->
           (py. model predict (-> test :text py/->py-list))
           py/->jvm
           first))]

    (-> test
          (ds/add-column :target prediction)
          (ds/select-columns [:id :target])
          (ds/write-csv! "kaggle_submission.csv"))))


(comment


  (def params
    (merge
     {:use_multiprocessing false
      :process_count 1
      :use_cuda true}

     (->
      (slurp "params.yaml")
      (yaml/parse-string)
      :train)))




  (def model
    (classification/ClassificationModel
     (:model_type params)
     "./outputs/best_model/"))

  (py/from-import lime.lime_text LimeTextExplainer)


  (def explainer (LimeTextExplainer :class_names ["fake" "real"]))
  (def numpy (py/import-module "numpy"))


  (py/from-import scipy.special softmax)
  (def predict-proba-fn

    (py/make-callable (fn [texts]
                        (let [soft-max
                              (->
                               (py. model predict (py/->py-list texts))
                               second
                               (softmax :axis -1))]

                          (py. numpy array (mapv first soft-max))))))


  (def test-ds (->
                (ds/dataset "test.csv" {:key-fn keyword})
                preprocess))

  (def train-ds (->
                 (ds/dataset "train.csv" {:key-fn keyword})
                 preprocess))

  (def real
    (-> train-ds
        (tc/select-rows   (fn [row] (= 1 (:labels row))))
        :text
        second))

  (def fake
    (-> train-ds
        (tc/select-rows   (fn [row] (= 0 (:labels row))))
        :text
        (nth 33)))


  (->
   (py/py. explainer explain_instance fake predict-proba-fn)
   (py/py. save_to_file "/tmp/fake.html"))

  (->
   (py/py. explainer explain_instance real predict-proba-fn)
   (py/py. save_to_file "/tmp/real.html"))







  (py/->jvm
   (py/cfn predict-proba-fn ["hello" "world"]))
                     
  (def builtins (py/import-module "builtins"))

  (py. builtins type xxx)

  (def main-globals (-> (py/add-module "__main__")
                        (py/module-dict)))

  (.put main-globals "xxx" xxx)

  (py/run-simple-string "print(xxx('hello'))")
  (py/run-simple-string "print(type(xxx))")

  :ok)
