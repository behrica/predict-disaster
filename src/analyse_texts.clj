(ns analyse-texts

  (:require
   ;; [scicloj.clay.v1.api :as clay]
   ;; [scicloj.clay.v1.tools :as tools]
   ;; [scicloj.clay.v1.extensions :as extensions]
   ;; [scicloj.clay.v1.tool.scittle :as scittle]
   ;; [scicloj.kindly.v2.api :as kindly]
   ;; [scicloj.kindly.v2.kind :as kind]
   ;; [scicloj.kindly.v2.kindness :as kindness]
   [nextjournal.clerk :as clerk]
   [nextjournal.clerk.viewer :as v]
   [tablecloth.api :as tc]
   [tech.v3.libs.arrow :as arrow]))

(comment
  (clerk/serve! {:browse? true}))
 

(clerk/add-viewers [{:pred tc/dataset?
                     :transform-fn #(clerk/table
                                     {:head (into [] (tc/column-names (v/->value %)))
                                      :rows (into [] (tc/rows (v/->value %) :as-seq))})}])

(def train
  (tc/dataset "train.csv" {:key-fn keyword}))


(tc/column-names train)

;; # Text length



(clerk/vl
 (-> train
     (tc/select-rows (fn [row] (= 0 (:target row))))
     (tc/add-column :text-len (fn [ds] (map count (:text ds))))
     (tc/rows :as-maps)
     ((fn [vals]
        {:data {:values vals}
         :layer [ {

                   :encoding {:x {:bin true :field :text-len}
                              :y {:aggregate "count"}}
                   :mark "bar"}
                 {:mark "rule"
                  :encoding {:x {"aggregate" "mean" "field" :text-len}
                             :color {:value :red}
                             :size {:value 5}}}]}))))


(clerk/vl
 (-> train
     (tc/select-rows (fn [row] (= 1 (:target row))))
     (tc/add-column :text-len (fn [ds] (map count (:text ds))))
     (tc/rows :as-maps)
     ((fn [vals]
        {:data {:values vals}
         :layer [ {

                   :encoding {:x {:bin true :field :text-len}
                              :y {:aggregate "count"}}
                   :mark "bar"}
                 {:mark "rule"
                  :encoding {:x {"aggregate" "mean" "field" :text-len}
                             :color {:value :red}
                             :size {:value 5}}}]}))))





(frequencies
 (:target train))

(frequencies
 (:location train))

(frequencies
 (:keyword train))

^{::clerk/visibility {:result :hide}}
(def keyword-vs-target
  (-> train
      (tc/drop-missing :keyword)
      (tc/group-by :keyword)
      (tc/drop-columns [:text :location])
      (tc/aggregate (fn [ds]
                      (let [n-rows (tc/row-count ds)
                            n-rows-1
                            (-> ds
                                (tc/select-rows (fn [row] (= 1 (:target row))))
                                (tc/row-count))]
                        {:n-rows n-rows
                         :perc-1 (/ n-rows-1 n-rows)})))))
      

(-> keyword-vs-target (tc/order-by :summary-perc-1 :desc) tc/head)

(-> keyword-vs-target (tc/order-by :summary-perc-1 :asc) tc/head)
