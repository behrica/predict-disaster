(ns simpletransformers
  (:require
   [scicloj.metamorph.ml :as ml]
   [tablecloth.api :as tc]
   [libpython-clj2.python.ffi :as ffi]
   [libpython-clj2.python :refer [py.- py.] :as py]))


(defn- train
  [feature-ds label-ds options]

  (let [
        model-args (:model-args options)
        _ (clojure.pprint/pprint model-args)
        pd (py/import-module "pandas")
        st (py/import-module "simpletransformers.classification")

        pd-train
        (-> feature-ds
            (tc/append label-ds)
            (tc/rows :as-seqs)
            ((py/py.- pd DataFrame)))

        eval-df
        (-> (:eval_df options)
            (tc/rows :as-seqs)
            ((py/py.- pd DataFrame)))

        model
        ( (py/py.- st ClassificationModel)
         (:model_type model-args)
         (:model_name model-args)

         :use_cuda (:use_cuda model-args)
         :args model-args)]
    (py/->jvm
     (py. model train_model pd-train :eval_df eval-df))))
      




    


   



(defn- predict
  [feature-ds thawed-model {:keys [target-columns target-categorical-maps options model-data] :as model}]
  (py/with-gil

    (let [

          pd (py/import-module "pandas")
          st (py/import-module "simpletransformers.classification")
          model
          ( (py/py.- st ClassificationModel)
           "electra" "outputs/best_model"
           :use_cuda true)]
      (->> (py. model predict (py/->py-list (get feature-ds :text)))
           (py/->jvm)
           first
           (hash-map :labels)
           (tc/dataset)))))

(ml/define-model! :simpletransformers/classification
  train
  predict
  {:documentation {:user-guide "http://simpletransformers.ai"
                   :javadoc ""}})
   ;; :options [{:name "batchsize" :type :int16 :default nil}
   ;;           {:name "model-spec" :type :model-spec :default nil}
   ;;           {:name "name" :type :string :default nil}
   ;;           {:name "initial-shape" :type :shape :default nil}
   ;;           {:name "nepoch" :type :int16 :default nil}]
