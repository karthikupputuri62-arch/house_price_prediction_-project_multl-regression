import os
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

class MINI:
    def __init__(self,path):
      try:
        self.path = path
        self.df = pd.read_csv(self.path)
        self.df['date']=pd.to_datetime(self.df['date'],dayfirst=True)
        self.df['day']=self.df['date'].dt.day
        self.df['month']=self.df['date'].dt.month # separate the date in columns
        self.df['year']=self.df['date'].dt.year
        self.df = self.df.drop(['date'],axis=1)
        self.df['country']=self.df['country'].map({'USA':0})
        self.df['city']=self.df['city'].map({'Shoreline':1, 'Seattle':2,'Kent':3,
                    'Bellevue':4,             'Redmond':5,        'Maple Valley':6,
                  'North Bend':7,    'Lake Forest Park':8,           'Sammamish':9,
                      'Auburn':10,          'Des Moines':11,             'Bothell':12,
                 'Federal Way':13,            'Kirkland':14,            'Issaquah':15,
                 'Woodinville':16,       'Normandy Park':17,           'Fall City':18,
                      'Renton':19,           'Carnation':20,          'Snoqualmie':21,
                      'Duvall':22,              'Burien':23,           'Covington':24,
         'Inglewood-Finn Hill':25,             'Kenmore':26,           'Newcastle':27,
               'Mercer Island':28,       'Black Diamond':29,          'Ravensdale':30,
                  'Clyde Hill':31,              'Algona':32,           'Skykomish':33,
                     'Tukwila':34,              'Vashon':35,        'Yarrow Point':36,
                      'SeaTac':37,              'Medina':38,            'Enumclaw':39,
             'Snoqualmie Pass':40,             'Pacific':41,  'Beaux Arts Village':42,
                     'Preston':43,              'Milton':44})
        self.X = self.df.iloc[:,1:]
        self.y = self.df.iloc[:,0]
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
      except Exception as e:
          err_type,err_msg,err_line=sys.exc_info()
          print(f"error from line no :{err_line.tb_lineno} due to {err_type} from {err_msg}")

    def model_prediction(self):

      try:
        self.reg = LinearRegression()
        self.reg.fit(self.X_train,self.y_train)
        # train predictions
        self.y_train_predictions = self.reg.predict(self.X_train)
        # test predictions
        self.y_test_predictions= self.reg.predict(self.X_test)
      except Exception as e:
          err_type,err_msg,err_line=sys.exc_info()
          print(f"error from line no :{err_line.tb_lineno} due to {err_type} from {err_msg}")

    def manul_accuracy(self,actual,predictions):

      try:
        # train accuracy
        total =0
        for i in range(len(actual)):
            total = total+actual.iloc[i] # actual.iloc [i]---actual.iloc[o]=0 is the index number
        mean = total/len(actual)

        numerator = 0
        for i in range(len(actual)): # length of rows
            error = actual.iloc[i]-predictions[i]  # actual.iloc [i]---actual.iloc[o]=0 is the index number
            numerator =numerator+( error ** 2)

        denominator = 0
        for i in range(len(actual)):
            square = actual.iloc[i]-mean
            denominator= denominator+( square ** 2)

        accuracy = 1-(numerator/denominator)
        return accuracy

      except Exception as e:
          err_type,err_msg,err_line=sys.exc_info()
          print(f"error from line no :{err_line.tb_lineno} due to {err_type} from {err_msg}")

    def manual_loss(self,actual,predictions):
     try:

        numerator = 0
        for i in range(len(actual)):
            loss= actual.iloc[i]-predictions[i]
            numerator=numerator+(loss**2)
        loss = numerator/len(actual-1)
        return loss

     except Exception as e:
         err_type, err_msg, err_line = sys.exc_info()
         print(f"error from line no :{err_line.tb_lineno} due to {err_type} from {err_msg}")

    def results(self):

     try:
        train_accuracy = self.manul_accuracy(self.y_train,self.y_train_predictions)
        test_accuracy = self.manul_accuracy(self.y_test,self.y_test_predictions)
        train_loss = self.manual_loss(self.y_train,self.y_train_predictions)
        test_loss = self.manual_loss(self.y_test,self.y_test_predictions)

        print(f'Train Accuracy: {train_accuracy}')
        print(f'Train loss: {train_loss}')
        print(f'Test Accuracy: {test_accuracy}')
        print(f'Test loss: {test_loss}')

     except Exception as e:
         err_type, err_msg, err_line = sys.exc_info()
         print(f"error from line no :{err_line.tb_lineno} due to {err_type} from {err_msg}")

    def save_model(self):

        try:

            with open("house.pkl", "wb") as f:

                pickle.dump(self.reg, f)

        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            print(f"error from line no :{err_line.tb_lineno} due to {err_type} from {err_msg}")



    def load_model(self):

        try:

            with open("house.pkl", "rb") as f:

                self.m = pickle.load(f)

        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            print(f"error from line no :{err_line.tb_lineno} due to {err_type} from {err_msg}")

    def sample(self,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated,city,country,day,month,year):
        try:
            result = self.m.predict([[bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated,city,country,day,month,year]])
            print(f'Result: {result[0]}')

        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            print(f"error from line no :{err_line.tb_lineno} due to {err_type} from {err_msg}")




if __name__ == '__main__':
    obj=MINI('data .csv')
    obj.model_prediction()
    obj.results()
    obj.save_model()
    obj.load_model()
    obj.sample(3,1.5,1340,7912,1.5,0,0,3,1340,0,1945,2005,2,0,22,4,2026)