#importing libs
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#open data set file as csv file
file_content = pd.read_csv ("breast-cancer.csv",usecols=["diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean"])

#concverting diagnoses to numbers to able to use in machine learning
file_content["diagnosis"] = file_content['diagnosis'].map({'M':0,'B':1})

#removing id data (not importat for machine learning model
file_content.pop('id')

#getting the output to test it
actual_diagnose  = file_content.pop('diagnosis')

#spletting 80% from data to test the model with it
input_train , input_test , output_train , output_test = train_test_split(file_content,actual_diagnose,test_size=0.2)

#making machine learning model
breast_cancer_model = LinearSVC()
breast_cancer_model.fit(input_train, output_train )
predicted_ouput  = breast_cancer_model.predict(input_test)

#model accuracy
preduction_accuracy = classification_report(output_test,predicted_ouput)
print( preduction_accuracy)















# import matplotlib.pyplot as plt
# #open data set as csv file
# file_content = pd.read_csv ("breast-cancer.csv",usecols=['area_mean', 'diagnosis','compactness_mean'])
#
# #checking if chossrn features is effected
# for index , dict_ in file_content.iterrows():
#     if dict_['diagnosis'] == 'M':
#         plt.scatter(dict_['area_mean'], dict_['compactness_mean'],c='b')
#     else:
#         plt.scatter(dict_['area_mean'], dict_['compactness_mean'],c='r')
#
# #plotting data
# plt.title("breast cancer")
# plt.show()
