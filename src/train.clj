(ns train
  (:require
   [scicloj.ml.dataset :as ds]
   [clojure.data.json :as json]
   [clj-yaml.core :as yaml]
   [tablecloth.api :as tc]
   [preprocess :refer [preprocess]]
   [simpletransformers]
   [scicloj.ml.core :as ml]
   [scicloj.ml.metamorph :as mm]
   [scicloj.ml.smile.nlp :as nlp]
   [tech.v3.libs.arrow :as arrow]
   [clojure.core.matrix.impl.pprint :as m-pprint]
   [confuse.multi-class-metrics :as confuse-mcm]
   [confuse.binary-class-metrics :as confuse-bcm]))


   
   
;; (py/initialize!)


(def params
  (->
   (slurp "params.yaml")
   (yaml/parse-string)
   :train))

(def pd-train
   (->
    (arrow/stream->dataset "train.arrow" {:key-fn keyword})
    (tc/select-columns [:text :labels])))



(def pd-eval
  (->
   (arrow/stream->dataset "test.arrow" {:key-fn keyword})
   (tc/select-columns [:text :labels])))



(defn my-text->bow [text options]
  (let [bow (nlp/default-text->bow text options)]
    (def bow bow)
    (->> bow
         (remove (fn [[term freq]]
                   (<  (count term) 3)))
         (into {}))))


(defn features [ds params]
  (-> ds
      (nlp/count-vectorize :text :bow {:stopwords :default
                                       :text->bow-fn my-text->bow})
      (nlp/bow->tfidf :bow :tfidf {:tf-map-handler-fn
                                   (partial nlp/tf-map-handler-top-n (:top-n params))})

      (nlp/tfidf->dense-array :tfidf :dense)
      (tc/drop-columns [:text])
      (tc/separate-column :dense (fn [array]

                                  (zipmap
                                   (map #(str "c-" %) (range (count array)))

                                   array)))))

(def data (tc/concat pd-train pd-eval))

(def data+features
  (-> data
   (tc/select-columns [:labels])
   (ds/categorical->number [:labels] [[0 0.0] [1 1.0]])
   (tc/append (features data params))))


(def train-ds
  (tc/head data+features (tc/row-count pd-train)))


(def eval-ds
  (tc/tail data+features (tc/row-count pd-eval)))

  

(let [pipe (ml/pipeline
            (mm/set-inference-target [:labels])
            (mm/model {:model-type :smile.classification/logistic-regression}))

      ctx-fit
      (ml/fit-pipe
       train-ds
       pipe)


      ctx-tf
      (ml/transform-pipe eval-ds pipe ctx-fit)


      actual (:labels pd-eval)
      predicted (-> ctx-tf :metamorph/data (ds/column-values->categorical :labels))

      _ (def actual actual)
      _ (def predicted predicted)
      mcc (confuse-mcm/multiclass-mcc actual predicted)]

 (->>
  (m-pprint/pm
   (->  (confuse-bcm/confusion-matrix actual predicted)
        (confuse-bcm/confusion-matrix-str)))
  (spit "validation-cm.txt"))

 (spit "eval.json"

       (json/write-str
        {:train {:mcc mcc}})))

(println "training finished")
(shutdown-agents)
