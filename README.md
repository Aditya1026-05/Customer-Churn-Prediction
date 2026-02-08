**Overview**

Customer churn is one of the biggest challenges for subscription-based and service businesses.
This project builds a predictive model that:

1. Identifies customers likely to churn

2. Assigns churn probabilities

3. Suggests personalized retention strategies

4. The system is deployed as a web application for real-time predictions.


**Machine Learning Pipeline**

1. Data preprocessing

      Handling Imbalanced data

      Encoding categorical variables

2. Model training

      Random Forest V/S Xg boost

3.Model evaluation

      Accuracy

      Confusion matrix

      ROC-AUC score

4. Model saving

   Trained model exported as .sav file




**Retention Strategy**


| Churn Probability | Risk Level | Strategy                    |
| ----------------- | ---------- | --------------------------- |
| < 0.3             | Low        | Loyalty rewards             |
| 0.3 – 0.7         | Medium     | Personalized offers         |
| > 0.7             | High       | Discount + priority support |











to use tel_churn.csv(one hot encoded data) we need to use ANN in model.ipynb and use compile method as 'categorical_crossentropy' and activation='softmax'(on last layer)

or to use first_telc.csv only on ANN use compile method as 'sparse_categorical_crossentropy' and activation='softmax'(on last layer)


we must choose sparse method
