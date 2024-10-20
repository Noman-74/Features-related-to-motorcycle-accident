import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.tree import plot_tree
import pickle

df=pd.read_csv('E:\\thesis\\N.csv')

df['bikersatt']=(df['Mobile / Streaming']+df['Drug Addiction ']+df['Overtaking ']+df[ 'Competitive riding ']+df['Traffic Law Disregard']+df['Inexperience ']+df['Overconfidence']+df['Overspeed ']+df['Panic Braking '])/9
def bike(df):
    if df['bikersatt']>0 and df['bikersatt']<=1:
        return 1
    elif df['bikersatt']>1 and df['bikersatt']<=2:
        return 2
    elif df['bikersatt']>2 and df['bikersatt']<=3:
        return 3
    elif df['bikersatt']>3 and df['bikersatt']<=4:
        return 4
    elif df['bikersatt']>4 and df['bikersatt']<=5:
        return 5
df['Bikers driving behavior']=df.apply(lambda df:bike(df),axis=1) 
df['condition']=(df['More Cc Bike']+df['Mechanical Problem']+df['Overloading']+df['Travel Distance '])/4
def m(df):
    if df['condition']>0 and df['condition']<=1:
        return 1
    elif df['condition']>1 and df['condition']<=2:
        return 2
    elif df['condition']>2 and df['condition']<=3:
        return 3
    elif df['condition']>3 and df['condition']<=4:
        return 4
    elif df['condition']>4 and df['condition']<=5:
        return 5
df['motorbike condition']=df.apply(lambda df:m(df),axis=1)
df['w']=(df['Rainy weather ']+df['Fog & dust']+df['High temperature '])/3
def n(df):
    if df['w']>0 and df['w']<=1:
        return 1
    elif df['w']>1 and df['w']<=2:
        return 2
    elif df['w']>2 and df['w']<=3:
        return 3
    elif df['w']>3 and df['w']<=4:
        return 4
    elif df['w']>4 and df['w']<=5:
        return 5
       
df['weather environment']=df.apply(lambda df:n(df),axis=1)
df['DE']=(df['Flyover, bridge, culvert']+df['Turn, bend, curve ']+df['Divider, Median, \nGuardrail  ']+df['Problematic \nCurb'])/4
def o(df):
    if df['DE']>0 and  df['DE']<=1:
        return 1
    elif df['DE']>1 and  df['DE']<=2:
        return 2
    elif df['DE']>2 and  df['DE']<=3:
        return 3
    elif df['DE']>3 and  df['DE']<=4:
        return 4
    elif df['DE']>4 and  df['DE']<=5:
        return 5
        
df['Driving Environment']=df.apply(lambda df:o(df),axis=1)
df['P']=(df['DistressDrainage ']+df['Level crossing \nproblems'])/2
def q(df):
    if df['P']>0 and df['P']<=1:
        return 1
    elif df['P']>1 and df['P']<=2:
        return 2
    elif df['P']>2 and df['P']<=3:
        return 3
    elif df['P']>3 and df['P']<=4:
        return 4
    elif df['P']>4 and df['P']<=5:
        return 5
        
        
df['Pavement condition']=df.apply(lambda df:q(df),axis=1)
df['S']=(df['Lighting problem ']+df['SignMarking'])/2
def a(df):
    if df['S']>0 and df['S']<=1:
        return 1
    elif df['S']>1 and df['S']<=2:
        return 2
    elif df['S']>2 and df['S']<=3:
        return 3
    elif df['S']>3 and df['S']<=4:
        return 4
    elif df['S']>4 and df['S']<=5:
        return 5
df['sign marking & lighting']=df.apply(lambda df:a(df),axis=1)
df['R']=(df['Law enforcement ']+df['Haphazard Parking, Stoppage']+df['Non lane  Heterogenous\n Traffic']+df['Intersection problem '])/4
def b(df):
    if df['R']>0 and df['R']<=1:
        return 1
    elif df['R']>1 and df['R']<=2:
        return 2
    elif df['R']>2 and df['R']<=3:
        return 3
    elif df['R']>3 and df['R']<=4:
        return 4
    elif df['R']>4 and df['R']<=5:
        return 5
      
df['traffic control & law']=df.apply(lambda df:b(df),axis=1)
df['M']=(df['Right Turn, \nmerge movements']+df['Frequent side \nroad entry ']+df['Cut in movement ']+df['Cut out movement ']+df['Two way \ntraffic movement '])/5
def c(df):
    if df['M']>0 and df['M']<=1:
        return 1
    elif df['M']>1 and df['M']<=2:
        return 2
    elif df['M']>2 and df['M']<=3:
        return 3
    elif df['M']>3 and df['M']<=4:
        return 4
    elif df['M']>4 and df['M']<=5:
        return 5
     
df['traffic movement']=df.apply(lambda df:c(df),axis=1)
df['Ped']=(df['Footpath Inadequacy\n']+df['Pedestrian along with traffic direction \n']+df['\nPedestrian facing with Traffic Direction\n']+df['Pedestrian crossing \n'])/4
def b(df):
    if df['Ped']>0 and df['Ped']<=1:
        return 1
    elif df['Ped']>1 and df['Ped']<=2:
        return 2
    elif df['Ped']>2 and df['Ped']<=3:
        return 3
    elif df['Ped']>3 and df['Ped']<=4:
        return 4
    elif df['Ped']>4 and df['Ped']<=5:
        return 5
      
df['Pedestrian activity']=df.apply(lambda df:b(df),axis=1)
df=df.drop(['bikersatt','condition','w','R','P','S','DE','M','Ped'],axis=1)
df=df.drop(['More Cc Bike', 'Mechanical Problem', 'Mobile / Streaming',
       'Overloading', 'Drug Addiction ', 'Overtaking ',
       'Traffic Law Disregard', 'Inexperience ', 'Overconfidence',
       'Overspeed ', 'Panic Braking ', 'Footpath Inadequacy\n',
       'Crash/Conflict Experience', 'Rainy weather ', 'Fog & dust',
       'High temperature ', 'DistressDrainage ', 'Lighting problem ',
       'Intersection problem ', 'Level crossing \nproblems',
       'Turn, bend, curve ', 'Flyover, bridge, culvert', 'SignMarking',
       'Divider, Median, \nGuardrail  ', 'Non lane  Heterogenous\n Traffic',
       'Haphazard Parking, Stoppage', 'Right Turn, \nmerge movements',
       'Frequent side \nroad entry ', 'Cut in movement ',
       'Two way \ntraffic movement ', 'Law enforcement ',
       'Pedestrian along with traffic direction \n', 'Pedestrian crossing \n',
       '\nPedestrian facing with Traffic Direction\n', 'Competitive riding ',
       'Travel Distance ', 'Unsmooth transitions\n of Flyovers',
       'Problematic \nCurb', 'Cut out movement '],axis=1)
x=df[['Perceived \nEndangerment ', 'Bikers driving behavior',
       'motorbike condition', 'weather environment',
       'Pavement condition', 'sign marking & lighting',
       'traffic control & law', 'traffic movement', 'Pedestrian activity']]
y=df['Driving Environment']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=3)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)
print(classification_report(y_test, y_pred))
print('\nAccuracy: {0:.4f}'.format(accuracy_score(y_test, y_pred)))
print('Confusion Matrix:\n',confusion_matrix(y_test, y_pred))

pickle.dump(rf, open('model_noman.pkl','wb'))

model = pickle.load(open('model_noman.pkl','rb'))

