**Overview**

Customer churn is a major challenge for telecom companies, where acquiring new customers is significantly more expensive than retaining existing ones.
This project builds a machine learning–based churn prediction system that not only predicts whether a customer will churn, but also provides:

1. Churn probability

2. Risk segmentation

3. Key churn drivers

4. Recommended retention campaigns

The goal is to transform raw customer data into actionable business insights that support targeted retention strategies.


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



**Dataset**

Dataset: Telco Customer Churn dataset

Domain: Telecommunications

Target Variable: Churn (Yes/No)



**Key Features:**

Customer tenure

Contract type

Monthly charges

Internet service type

Payment method

Add-on services


**Exploratory Data Analysis (EDA)**

EDA was performed to understand churn behavior and feature relationships.

Key Insights:

1. Customers on month-to-month contracts show higher churn rates.

2. Low-tenure customers are more likely to churn.

3. Higher monthly charges correlate with increased churn risk.

4. Customers without add-on services tend to churn more.







to use tel_churn.csv(one hot encoded data) we need to use ANN in model.ipynb and use compile method as 'categorical_crossentropy' and activation='softmax'(on last layer)

or to use first_telc.csv only on ANN use compile method as 'sparse_categorical_crossentropy' and activation='softmax'(on last layer)


we must choose sparse method
