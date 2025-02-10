# Gestational Diabetes Prediction 
This study was conduct based on the data from [UCI Machine Learning](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?select=diabetes.csv) and aim to compare some algorithms for classifying the data. 

**Study Objective** : Developing Machine Learning model by Comparing some classification algorithms for detecting gestational diabetes with high recall to minimize miss treatment for patient

In this study, I try to make some scenarios to compare the algorithms, here are the scenarios:
- Modeling and comparing algorithms for Imbalance Class 
- Modeling and comparing algorithms for Balance Class
- Comparing the best algorithm from 2 scenarios above

Here are the outline of this project and the insights found in every section:

## A. Exploratory Data Analysis

This step will help for understanding more about the structure, patterns, amd relationships in dataset. 
This step is divided into some sections and for every section here I provide the insight I found
### Data Overview 
#### Insight
- Number of Column : 9
- Number of Entries : 786 
- Number of Missing Values : 0
- Name of Column:
    - `num_pregnant`            : the number of pregnancies the individual has had         
    - `glucose_concentration`   : the concentration of glucose in the individual's blood
    - `blood_pressure`          : the individual's blood pressure reading 
    - `triceps_thickness`       : the thickness of the individual's triceps skinfold, a measure of body fat
    - `two_hour_insulin`        : Insulin level in the individual's blood measured after 2 hours glucose test
    - `bmi`                     : the individual's Body Mass Index, measure of body fat based on height and weight
    - `pedigree_function`       : likelihood of the individual having diabetes based in familiy history
    - `age`                     : the individual's age
    - `is_diabetes`             : the target varibale, 1 indicating that the individual has diabetes, 0 otherwise. 

### Summary Statistics
This step helps me with brief understanding of the distribution, central tendency, and spread of the data
#### Insight
- `two_hour_insulin` feature has highest standard deviasion and this indicate that this feature has high variability or dispersion due to data points are widely spread out from the mean.
- `pedigree_function` feature has the lowest standard deviasion and this indicate that this feature has low variability or consistency in the data
- Feature such as `glucose_concentration`, `blood_pressure`, `triceps_thickness`, `two_hour_insulin`, `bmi` **cannot have** a value of **0**, this lead to replacing the 0 value in next preprocessing step 

### Distribution Analysis 
This step helps me to know more about the data distribution, in this step I tried to visualize the data 
#### Insight

**Univariate Analysis**

*Feature Distribution*

- `num_pregnant`, `predigree_function`, `two_hour_insulin`, `triceps_thickness`, and `age` features are not distributed normally and had right skwed distribution 
- `BMI`, `blood_pressure`, and `glucose_concentration` distributed normally

![alt text](assets/data_dist.png)
 
*Class Distribution*
- The data has 2 classes, 1 indicating that the individual has diabetes, 0 otherwise
- The class ditribution is Imbalance which is class 0 has more data compare to class 1 and the ratio is 2:1 

![alt text](assets/class_dist.png)

**Bivariate Analysis**

- `num_pregnant`: Judging from the boxplot, mothers who are pregnant 1-5 times have the same chance of experiencing diabetes, but mothers who are pregnant >5 times are more likely to suffer from diabetes than those who are pregnant <5 times.
- `glucose_concentration`: People with glucose concentration >125 mg/dL are more likely to experience gestational diabetes compared to people with glucose concentration <125 mg/dL.
- `blood_pressure`: Judging from the boxplot, there is no significant difference in the number of pregnant women with or without diabetes based on blood pressure.
- `triceps_thickness`: Judging from the boxplot, there is no significant difference in the number of pregnant women with or without diabetes based on triceps_thickness.
- `BMI`: Pregnant women with diabetes tend to have a BMI >35
- `age`: Pregnant women with diabetes are mostly between 38-45 years old

![alt text](assets/Bi_var.png)

**Multivariate Analysis**

- Judging from the heat_map, all features have a positive correlation with the target
- `glucose_concentration` is the feature with the highest correlation to the target, which is 0.47, but this correlation value does not show a strong level of correlation, as well as with other features
- There is no correlation between features that is that strong (>0.5)

![alt text](assets/heat_map.png)

### EDA Insight 
- The features `glucose_concentration`, `blood_pressure`, `triceps_thickness`, `two_hour_insulin`, `bmi` cannot have a value of 0, so it is necessary to replace the value 0 with NaN
- To ensure the causal relationship between features and between features and targets, a hypothesis test can be carried out
- Some features, such as `num_pregnant`, `predigree_function`, `two_hour_insulin`, `triceps_thickness`, and `age` had skewed distribution that may lead to data outlier. Thus, I consider to add handling outliers in the preprocessing section 

## Data Preprocessing 

- I tried to replace the features that can not have 0 value with NaN
- All missing values are imputed by the feature median
- The outliers are handled by using z-score 
- I did standardize all features using `StandardScaler()` 
- In EDA section, I mentioned that the target or class has imbalance value, which the ratio between class 0 and 1 is 2:1. In this step, I create new variables, `X_balanced`, and `y_balanced` to store the features and target with balanced class while the Imbalanced data are stored in the `X` and `y`. 
- After dividing the data into blanced and imbalanced, I used train-test split for eacj data and divided them into 80% data training and 20% data testing. 

## Modeling
The target indicates the prediction of whether a person has diabetes or not based on the features, this kind of prediction is a binary classification problem where the target variable (is_diabetes) is either 0 (no diabetes) or 1 (diabetes). Here are some models that may suit with this kind of classification:
- Logistic Regression
- Decision Tree
- Random Fores
- SVM
- KNN
- Neural Network

In this section, I divided the process into 3 scenarios
- First, modeling the Imbalanced data
- Second, modeling the balanced data
- Comparing all models from 2 scenarios 

## Modeling Imbalanced Class 

### Evaluating Model on Training Set

| Model                        | Accuracy | Precision | Recall | ROC_AUC |
|------------------------------|----------|-----------|--------|---------|
| Logistic Regression          | 0.77     | 0.69      | 0.57   | 0.85    |
| Decision Tree                | 1.00     | 1.00      | 1.00   | 1.00    |
| K-Nearest Neighbors          | 0.84     | 0.79      | 0.72   | 0.91    |
| Support Vector Classifier    | 0.80     | 0.77      | 0.60   | 0.90    |
| Random Forest Classifier     | 1.00     | 1.00      | 1.00   | 1.00    |
| Gradient Boosting Classifier | 0.91     | 0.90      | 0.82   | 0.98    |
| Neaural Network              | 0.81     | 0.77      | 0.65   | 0.90    |

### Evaluating Model on Test Set

| Model                        | Accuracy | Precision | Recall | ROC_AUC |
|------------------------------|----------|-----------|--------|---------|
| Logistic Regression          | 0.72     | 0.72      | 0.49   | 0.83    |
| Decision Tree                | 0.64     | 0.57      | 0.58   | 0.63    |
| K-Nearest Neighbors          | 0.70     | 0.65      | 0.52   | 0.79    |
| Support Vector Classifier    | 0.70     | 0.70      | 0.44   | 0.82    |
| Random Forest Classifier     | 0.72     | 0.70      | 0.52   | 0.82    |
| Gradient Boosting Classifier | 0.73     | 0.72      | 0.52   | 0.84    |
| Neaural Network              | 0.72     | 0.71      | 0.54   | 0.84    |

### Cross Valuidation 
| Model                        | mean_score | std_score |
|------------------------------|------------|-----------|
| Logistic Regression          | 0.57       | 0.02      | 
| Decision Tree                | 0.54       | 0.08      | 
| K-Nearest Neighbors          | 0.59       | 0.033      | 
| Support Vector Classifier    | 0.56       | 0.032      | 
| Random Forest Classifier     | 0.58       | 0.07      | 
| Gradient Boosting Classifier | 0.60       | 0.05      |
| Neaural Network              | 0.61       | 0.06      | 

### Insight
Due to the data that used in this section is imbalanced data, the evaluation results that will be highlighted are the ROC_AUC and recall values as well as cross validation to compare model consistency.

**Based on ROC_AUC and Recall**

- For the evaluation results using Training data, the `Decision Tree` and `Random Forest` models provide the highest ROC_AUC, accuracy, and recall of 1, but when evaluated with test data, the accuracy and recall for each model show an overfitting model, where the evaluation results with test data differ by almost 50%
- All models show overfitting by comparing accuracy, recall, and ROC_AUC between the train set and test set

**Based on Cross Validation**

- Judging from the cross validation results, the `Logistic Regression`, `DC` and `SVM` models have low mean and standard deviation values, indicating that the model may be more reliable 
- The `GBC`, `RF`, and `NN` models have high mean and standard deviation values, which indicates that the model may be strong but inconsistent. Compared to all models, 
- The `KNN` model has a fairly high mean value and a fairly low standard deviation indicating that the model may be strong and consistent.

## Modeling Balanced Class 

### Evaluating Model on Training Set

| Model                        | Accuracy | Precision | Recall | ROC_AUC |
|------------------------------|----------|-----------|--------|---------|
| Logistic Regression          | 0.74     | 0.75      | 0.74   | 0.85    |
| Decision Tree                | 1.00     | 1.00      | 1.00   | 1.00    |
| K-Nearest Neighbors          | 0.85     | 0.81      | 0.91   | 0.94    |
| Support Vector Classifier    | 0.83     | 0.81      | 0.87   | 0.91    |
| Random Forest Classifier     | 1.00     | 1.00      | 1.00   | 1.00    |
| Gradient Boosting Classifier | 0.93     | 0.92      | 0.94   | 0.98    |
| Neaural Network              | 0.84     | 0.81      | 0.88   | 0.91    |

### Evaluating Model on Test Set

| Model                        | Accuracy | Precision | Recall | ROC_AUC |
|------------------------------|----------|-----------|--------|---------|
| Logistic Regression          | 0.76     | 0.76      | 0.75   | 0.85    |
| Decision Tree                | 0.78     | 0.76      | 0.82   | 0.76    |
| K-Nearest Neighbors          | 0.79     | 0.75      | 0.86   | 0.85    |
| Support Vector Classifier    | 0.83     | 0.80      | 0.87   | 0.90    |
| Random Forest Classifier     | 0.83     | 0.81      | 0.86   | 0.90    |
| Gradient Boosting Classifier | 0.79     | 0.78      | 0.81   | 0.88    |
| Neaural Network              | 0.81     | 0.78      | 0.86   | 0.89    |

### Cross Valuidation 
| Model                        | mean_score | std_score |
|------------------------------|------------|-----------|
| Logistic Regression          | 0.73       | 0.03      | 
| Decision Tree                | 0.79       | 0.07      | 
| K-Nearest Neighbors          | 0.89       | 0.06      | 
| Support Vector Classifier    | 0.85       | 0.05      | 
| Random Forest Classifier     | 0.84       | 0.09      | 
| Gradient Boosting Classifier | 0.83       | 0.07      |
| Neaural Network              | 0.84       | 0.05      | 

### Insight
Because the data used is balanced, the evaluation results that will be highlighted are the accuracy and recall values, as well as cross validation to compare model consistency.

**Based on accuracy**
- All models show overfitting and underfitting seen from the results of the evaluation of train data and test data.

- `LR` is the only model that shows higher train data accuracy results compared to test data, indicating overfitting
- Models other than `LR` have higher test data accuracy compared to train data, indicating underfitting

**Based on recall**
- Models such as `DC`, `KNN`, `RF` and `GBC` show overfitting where the recall results using train data are higher than test data, with a fairly large difference in results
- `LR`, `SVM`, and `NN` show evaluation results that are not significantly different between train and test data
- `SVM` is a model that almost gives the same recall value between train and test data

**Based on Cross Validation**
- The `LR` model has a low mean accuracy and standard deviation value, indicating that the model may be more reliable but not powerful
- The `SVM` model has a high accuracy value and low standard deviation, indicating that the model may be more reliable and may be strong
- Models other than `LR` and `SVM` have high accuracy values ​​but their standard deviations are also high, indicating that the model may be strong but inconsistent.

## Performance Analysis

The models generated from balanced data showed better performance, with the `SVM` model providing 87% recall and more consistent and robust cross validation results.

## Future Consideration
- **Feature Engineering:** Further refinement of features and potential addition of new features could enhance model performance.
- **Hyperparameter Tuning:** Fine-tuning model parameters might improve ROC_AUC, accuracy, and recall.
- **Alternative Resampling Techniques:** Exploring other techniques like ADASYN or different sampling strategies could provide better results than SMOTE.