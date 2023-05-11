import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
import sklearn.model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('Customers-2.csv')


def displayHeadRows(df, rows = 10):
    return df.head(rows)



for col in df:
    if df[col].dtype =='object':
      df[col]=OrdinalEncoder().fit_transform(df[col].values.reshape(-1,1))

class_label =df['Spending Score (1-100)']
df = df.drop(['Spending Score (1-100)'], axis =1)
df = (df-df.min())/(df.max()-df.min())
# # add back the original target column
df['Spending Score (1-100)']=class_label

#pre-processing
# continuous vs ordinal(0-1) so the data looks all similar
# do not want to lose the original df
customer_data = df.copy()
le = preprocessing.LabelEncoder()
age = le.fit_transform(list(customer_data["Age"])) # age in years
gender = le.fit_transform(list(customer_data["Gender"])) # gender (0 = male; 1 = female)
CustomerID = le.fit_transform(list(customer_data["CustomerID"]))
experience = le.fit_transform(list(customer_data["Work Experience"]))
Profession = le.fit_transform(list(customer_data["Profession"]))
income = le.fit_transform(list(customer_data["Annual Income ($)"]))
family = le.fit_transform(list(customer_data["Family Size"]))
score = le.fit_transform(list(customer_data["Spending Score (1-100)"]))

#x=input, y=output converted into list
x = list(zip(age, gender, CustomerID, gender, experience, Profession, income, family))
#output=classified attribute
y = list(score)
# Test options and evaluation metric
#converts the training data set in to batches
num_folds = 5 #no. of batches
seed = 7 #uses different combination randomly
scoring = 'accuracy' #metric

# Model Test/Train
# Splitting what we are trying to predict into 4 different arrays -
# X train is a section of the x array(attributes) and vise versa for Y(features)
# The test data will test the accuracy of the model created
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=42)
#splitting 20% of our data into test samples. If we train the model with higher data it already has seen that information and knows

#size of train and test subsets after splitting
def test_train(df):
    print(np.shape(x_train), np.shape(x_test))
    return np.shape(x_train), np.shape(x_test)

# Predictive analytics model development by comparing different Scikit-learn classification algorithms
# identify the best performing model listing the classifiers
def classifiers(df):
    models = []
    models.append(('DT', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('RF', RandomForestClassifier()))

    results = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
        results.append({'Model': name, 'Accuracy': cv_results.mean(), 'Standard Deviation': cv_results.std()})

    df_results = pd.DataFrame(results)
    return df_results

    # Compare Algorithms' Performance
    # box plot to visually view classifiers
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)

    models.append(('DT', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('RF', RandomForestClassifier()))
    dt = DecisionTreeClassifier()
    nb = GaussianNB()
    gb = GradientBoostingClassifier()
    rf = RandomForestClassifier()

    best_model = dt
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    print("Best Model Accuracy Score on Test Set:", accuracy_score(y_test, y_pred))

    #Model Performance Evaluation Metric 2
    #Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    return fig


def predictions(df):
    best_model = DecisionTreeClassifier()
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)

    pred_df = pd.DataFrame(columns=['Predicted', 'Actual', 'Data'])
    for i in range(len(y_pred)):
        pred_df.loc[i] = [y_pred[i], y_test[i], x_test[i]]

    return pred_df