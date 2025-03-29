# Predicting Bank Customer Churn using Support Vector Machine Model

# **Introduction**

The aim of this project is to predict customer churn in a bank using machine learning techniques. Customer churn is a critical issue for banks as retaining customers is more cost-effective than acquiring new ones. By accurately predicting which customers are likely to leave, banks can take proactive measures to improve customer retention. We employ Support Vector Machine (SVM) models due to their effectiveness in classification tasks.

# **Process**

**Data Preprocessing**


*   **Loading Data:** The dataset is loaded and initial exploration is conducted to understand its structure.
*   **Handling Missing Values:** We check for any missing values and handle them appropriately to ensure data quality.

*   **Encoding Categorical Variables:** Categorical variables such as 'Geography' and 'Gender' are encoded to numerical values for model compatibility.
*   **Balancing the Dataset:** Since customer churn data is often imbalanced, we apply under-sampling and over-sampling techniques to balance the class distribution.

*   **Feature Scaling:** Continuous variables are standardized to improve model performance.

# **Models Used**

* **Original SVM Model:** The baseline SVM model trained on the original, imbalanced dataset.
* **Under-sampled SVM Model:** SVM model trained on a dataset where the majority class is under-sampled to balance the class distribution.
* **Over-sampled SVM Model:** SVM model trained on a dataset where the minority class is over-sampled to balance the class distribution.
* **Tuned SVM Models:** SVM models with hyperparameter optimization applied, trained on the original, under-sampled, and over-sampled datasets. Hyperparameters such as C, gamma, and class_weight are tuned using GridSearchCV to improve model performance.


# **Final Results**

Among all the models, the Tuned SVM with over-sampled data achieved the highest accuracy of **0.93**. This model outperformed others due to:

**Hyperparameter Optimization:** Fine-tuning the model parameters (C, gamma, class_weight) allowed for better generalization and performance on unseen data.
Over-sampling: Balancing the class distribution helped the model better handle the minority class, leading to improved accuracy.


# **Conclusion**
This project demonstrates the importance of data preprocessing, handling class imbalance, and hyperparameter tuning in building effective machine learning models. The Tuned SVM with over-sampled data provided the best performance, showcasing the potential of advanced techniques in predictive analytics for customer churn. By implementing these strategies, banks can better predict and mitigate customer churn, leading to improved customer retention and business outcomes.
