# Titanic-predication-
Predicate the survival rate of passengers by machine learning alogarithm





TITANIC SURVIVAL PREDICATION
The titanic dataset provides information on the fate of the passengers on the titanic, summarized according to economic status ,sex,age,and survival.

OUR TASK IS TO PREDICT THE SURVIVAL OF THE TITANIC PASSENGERS 


                                                                              --LOGISTIC REGRESSION&CORRELATION
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
%matplotlib inline                       

                            #IMPORTING COMMON LIBRARIES USED FOR DATA CLEANING AND DATA VISUALIZATION
import seaborn as sns
sns.set(style="dark",color_codes=True)
sns.set(font_scale=1.5)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
                             ## IMPORTING MACHINE LEARNING ALOGARITHMS
LOAD THE DATA SET TO JUPYTER NOTEBOOK
import os
import pandas as pd
os.getcwd()
'C:\\Users\\ramth'
os.chdir("C:\\Users\\ramth\\OneDrive\\Desktop")
jp_train=pd.read_csv("train.csv")
                                                                    #  READ THE CSV FILE PROCESS
jp_train
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	A/5 21171	7.2500	NaN	S
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C
2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S
3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	113803	53.1000	C123	S
4	5	0	3	Allen, Mr. William Henry	male	35.0	0	0	373450	8.0500	NaN	S
...	...	...	...	...	...	...	...	...	...	...	...	...
886	887	0	2	Montvila, Rev. Juozas	male	27.0	0	0	211536	13.0000	NaN	S
887	888	1	1	Graham, Miss. Margaret Edith	female	19.0	0	0	112053	30.0000	B42	S
888	889	0	3	Johnston, Miss. Catherine Helen "Carrie"	female	NaN	1	2	W./C. 6607	23.4500	NaN	S
889	890	1	1	Behr, Mr. Karl Howell	male	26.0	0	0	111369	30.0000	C148	C
890	891	0	3	Dooley, Mr. Patrick	male	32.0	0	0	370376	7.7500	NaN	Q
891 rows × 12 columns

Understanding The Data set:-
1.SIBSP= Relatives of the passengers

2.PCLASS= Passenger class

3.EMBARKED= It implies where the traveller mounted from {s-southampton 70% Q-Queenstown 10% C-Cherbourg 20%}

4.PARCH= No.of parents /children

 jp_train.head(69)
PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Fare
0	1	0	3	1	22.0	1	0	7.2500
1	2	1	1	0	38.0	1	0	71.2833
2	3	1	3	0	26.0	0	0	7.9250
3	4	1	1	0	35.0	1	0	53.1000
4	5	0	3	1	35.0	0	0	8.0500
...	...	...	...	...	...	...	...	...
64	65	0	1	1	28.0	0	0	27.7208
65	66	1	3	1	28.0	1	1	15.2458
66	67	1	2	0	29.0	0	0	10.5000
67	68	0	3	1	19.0	0	0	8.1583
68	69	1	3	0	17.0	4	2	7.9250
69 rows × 8 columns

Start with Data Exploration&Cleaning
jp_train.shape
(891, 9)
jp_train.isnull().sum()                                                    #CHECKING NULLVALUES in data set
PassengerId      0
Survived         0
Pclass           0
Sex              0
Age            177
SibSp            0
Parch            0
Fare             0
Embarked         2
dtype: int64
jp_train.describe()
PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Fare
count	891.000000	891.000000	891.000000	891.000000	714.000000	891.000000	891.000000	891.000000
mean	446.000000	0.383838	2.308642	0.647587	29.699118	0.523008	0.381594	32.204208
std	257.353842	0.486592	0.836071	0.477990	14.526497	1.102743	0.806057	49.693429
min	1.000000	0.000000	1.000000	0.000000	0.420000	0.000000	0.000000	0.000000
25%	223.500000	0.000000	2.000000	0.000000	20.125000	0.000000	0.000000	7.910400
50%	446.000000	0.000000	3.000000	1.000000	28.000000	0.000000	0.000000	14.454200
75%	668.500000	1.000000	3.000000	1.000000	38.000000	1.000000	0.000000	31.000000
max	891.000000	1.000000	3.000000	1.000000	80.000000	8.000000	6.000000	512.329200
jp_train.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 9 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Sex          891 non-null    int64  
 4   Age          714 non-null    float64
 5   SibSp        891 non-null    int64  
 6   Parch        891 non-null    int64  
 7   Fare         891 non-null    float64
 8   Embarked     889 non-null    object 
dtypes: float64(2), int64(6), object(1)
memory usage: 62.8+ KB
jp_train.dtypes
PassengerId      int64
Survived         int64
Pclass           int64
Sex              int64
Age            float64
SibSp            int64
Parch            int64
Fare           float64
Embarked        object
dtype: object
jp_train.Sex.value_counts()         
1    577
0    314
Name: Sex, dtype: int64
jp_train
PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Fare	Embarked
0	1	0	3	1	22.0	1	0	7.2500	S
1	2	1	1	0	38.0	1	0	71.2833	C
2	3	1	3	0	26.0	0	0	7.9250	S
3	4	1	1	0	35.0	1	0	53.1000	S
4	5	0	3	1	35.0	0	0	8.0500	S
...	...	...	...	...	...	...	...	...	...
886	887	0	2	1	27.0	0	0	13.0000	S
887	888	1	1	0	19.0	0	0	30.0000	S
888	889	0	3	0	NaN	1	2	23.4500	S
889	890	1	1	1	26.0	0	0	30.0000	C
890	891	0	3	1	32.0	0	0	7.7500	Q
891 rows × 9 columns

from sklearn import preprocessing                                                       #import label encoder
label_encoder=preprocessing.LabelEncoder()                
jp_train['Sex']= label_encoder.fit_transform(jp_train['Sex'])
jp_train['Sex'].value_counts()
jp_train                                                                                                  #male=1,female=2
PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Fare	Embarked
0	1	0	3	1	22.0	1	0	7.2500	S
1	2	1	1	0	38.0	1	0	71.2833	C
2	3	1	3	0	26.0	0	0	7.9250	S
3	4	1	1	0	35.0	1	0	53.1000	S
4	5	0	3	1	35.0	0	0	8.0500	S
...	...	...	...	...	...	...	...	...	...
886	887	0	2	1	27.0	0	0	13.0000	S
887	888	1	1	0	19.0	0	0	30.0000	S
888	889	0	3	0	NaN	1	2	23.4500	S
889	890	1	1	1	26.0	0	0	30.0000	C
890	891	0	3	1	32.0	0	0	7.7500	Q
891 rows × 9 columns

jp_train.Embarked.value_counts()
S    644
C    168
Q     77
Name: Embarked, dtype: int64
jp_train.Survived.value_counts()
0    549
1    342
Name: Survived, dtype: int64
jp_train                                                                                                   #1016.69
PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Fare
0	1	0	3	1	22.0	1	0	7.2500
1	2	1	1	0	38.0	1	0	71.2833
2	3	1	3	0	26.0	0	0	7.9250
3	4	1	1	0	35.0	1	0	53.1000
4	5	0	3	1	35.0	0	0	8.0500
...	...	...	...	...	...	...	...	...
886	887	0	2	1	27.0	0	0	13.0000
887	888	1	1	0	19.0	0	0	30.0000
888	889	0	3	0	28.0	1	2	23.4500
889	890	1	1	1	26.0	0	0	30.0000
890	891	0	3	1	32.0	0	0	7.7500
891 rows × 8 columns

jp_train.head()
PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Fare
0	1	0	3	1	22.0	1	0	7.2500
1	2	1	1	0	38.0	1	0	71.2833
2	3	1	3	0	26.0	0	0	7.9250
3	4	1	1	0	35.0	1	0	53.1000
4	5	0	3	1	35.0	0	0	8.0500
jp_train.isnull().sum()
PassengerId    0
Survived       0
Pclass         0
Sex            0
Age            0
SibSp          0
Parch          0
Fare           0
dtype: int64
jp_train['Age'].median()
28.0
jp_train['Age']=jp_train['Age'].fillna(value=28)
jp_train
PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Fare	Embarked
0	1	0	3	1	22.0	1	0	7.2500	S
1	2	1	1	0	38.0	1	0	71.2833	C
2	3	1	3	0	26.0	0	0	7.9250	S
3	4	1	1	0	35.0	1	0	53.1000	S
4	5	0	3	1	35.0	0	0	8.0500	S
...	...	...	...	...	...	...	...	...	...
886	887	0	2	1	27.0	0	0	13.0000	S
887	888	1	1	0	19.0	0	0	30.0000	S
888	889	0	3	0	28.0	1	2	23.4500	S
889	890	1	1	1	26.0	0	0	30.0000	C
890	891	0	3	1	32.0	0	0	7.7500	Q
891 rows × 9 columns

jp_train['Age'].isnull().sum()
0
jp_train.Embarked.value_counts()
S    644
C    168
Q     77
Name: Embarked, dtype: int64
jp_train['Embarked']=jp_train['Embarked'].fillna(value='s')
                                            
jp_train
PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Fare	Embarked
0	1	0	3	1	22.0	1	0	7.2500	S
1	2	1	1	0	38.0	1	0	71.2833	C
2	3	1	3	0	26.0	0	0	7.9250	S
3	4	1	1	0	35.0	1	0	53.1000	S
4	5	0	3	1	35.0	0	0	8.0500	S
...	...	...	...	...	...	...	...	...	...
886	887	0	2	1	27.0	0	0	13.0000	S
887	888	1	1	0	19.0	0	0	30.0000	S
888	889	0	3	0	28.0	1	2	23.4500	S
889	890	1	1	1	26.0	0	0	30.0000	C
890	891	0	3	1	32.0	0	0	7.7500	Q
891 rows × 9 columns

jp_train['Embarked'].isnull().sum()
0
jp_train
PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Fare	Embarked
0	1	0	3	1	22.0	1	0	7.2500	S
1	2	1	1	0	38.0	1	0	71.2833	C
2	3	1	3	0	26.0	0	0	7.9250	S
3	4	1	1	0	35.0	1	0	53.1000	S
4	5	0	3	1	35.0	0	0	8.0500	S
...	...	...	...	...	...	...	...	...	...
886	887	0	2	1	27.0	0	0	13.0000	S
887	888	1	1	0	19.0	0	0	30.0000	S
888	889	0	3	0	28.0	1	2	23.4500	S
889	890	1	1	1	26.0	0	0	30.0000	C
890	891	0	3	1	32.0	0	0	7.7500	Q
891 rows × 9 columns

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
jp_train['Embarked']= label_encoder.fit_transform(jp_train['Embarked'])
jp_train['Embarked'].value_counts()
2    644
0    168
1     77
3      2
Name: Embarked, dtype: int64
jp_train
PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Fare	Embarked
0	1	0	3	1	22.0	1	0	7.2500	2
1	2	1	1	0	38.0	1	0	71.2833	0
2	3	1	3	0	26.0	0	0	7.9250	2
3	4	1	1	0	35.0	1	0	53.1000	2
4	5	0	3	1	35.0	0	0	8.0500	2
...	...	...	...	...	...	...	...	...	...
886	887	0	2	1	27.0	0	0	13.0000	2
887	888	1	1	0	19.0	0	0	30.0000	2
888	889	0	3	0	28.0	1	2	23.4500	2
889	890	1	1	1	26.0	0	0	30.0000	0
890	891	0	3	1	32.0	0	0	7.7500	1
891 rows × 9 columns

DATA AND INFO VISUALIZATION
import seaborn as sns
sns.countplot(jp_train['Embarked'],hue=jp_train['Survived'])
C:\Users\ramth\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<AxesSubplot:xlabel='Embarked', ylabel='count'>

jp_train.corr()                                             #KARL PEARSON'S COEFFIECENT OF CORRELATION
PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Fare
PassengerId	1.000000	-0.005007	-0.035144	0.042939	0.034212	-0.057527	-0.001652	0.012658
Survived	-0.005007	1.000000	-0.338481	-0.543351	-0.064910	-0.035322	0.081629	0.257307
Pclass	-0.035144	-0.338481	1.000000	0.131900	-0.339898	0.083081	0.018443	-0.549500
Sex	0.042939	-0.543351	0.131900	1.000000	0.081163	-0.114631	-0.245489	-0.182333
Age	0.034212	-0.064910	-0.339898	0.081163	1.000000	-0.233296	-0.172482	0.096688
SibSp	-0.057527	-0.035322	0.083081	-0.114631	-0.233296	1.000000	0.414838	0.159651
Parch	-0.001652	0.081629	0.018443	-0.245489	-0.172482	0.414838	1.000000	0.216225
Fare	0.012658	0.257307	-0.549500	-0.182333	0.096688	0.159651	0.216225	1.000000
Barplot -To shows the relationship b/w NUMERIC & CATEGORIC VALUE
jp_train.plot(x="Survived",y=["SibSp","Parch"],kind='bar')
<AxesSubplot:xlabel='Survived'>

correlation=jp_train.corr()
correlation['Survived'].sort_values(ascending=False)
Survived       1.000000
Fare           0.257307
Parch          0.081629
PassengerId   -0.005007
SibSp         -0.035322
Age           -0.064910
Embarked      -0.163517
Pclass        -0.338481
Sex           -0.543351
Name: Survived, dtype: float64
HEAT MAP----To visualize the Density
sns.heatmap(jp_train.corr())
<AxesSubplot:>

correlation['Fare'].sort_values(ascending=False)
correlation['Fare']
PassengerId    0.012658
Survived       0.257307
Pclass        -0.549500
Sex           -0.182333
Age            0.096688
SibSp          0.159651
Parch          0.216225
Fare           1.000000
Embarked      -0.221226
Name: Fare, dtype: float64
jp_train['family']=jp_train['SibSp']+jp_train['Parch']+1
jp_train=jp_train.drop(['SibSp','Parch'],axis=1)
jp_train
jp_train=jp_train.drop(['Embarked'],axis=1)
jp_train                                                                                                 #END OF EDA
PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Fare
0	1	0	3	1	22.0	1	0	7.2500
1	2	1	1	0	38.0	1	0	71.2833
2	3	1	3	0	26.0	0	0	7.9250
3	4	1	1	0	35.0	1	0	53.1000
4	5	0	3	1	35.0	0	0	8.0500
...	...	...	...	...	...	...	...	...
886	887	0	2	1	27.0	0	0	13.0000
887	888	1	1	0	19.0	0	0	30.0000
888	889	0	3	0	28.0	1	2	23.4500
889	890	1	1	1	26.0	0	0	30.0000
890	891	0	3	1	32.0	0	0	7.7500
891 rows × 8 columns

APPLY LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x=jp_train.drop("Survived",axis=1).values
y=jp_train['Survived'].values
TRAINING AND TESTING THE DATA
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
(623, 8)
(268, 8)
(623,)
(268,)
LogReg=LogisticRegression()
LogReg.fit(x_train,y_train)
C:\Users\ramth\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
LogisticRegression()
y_pred=LogReg.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
0.7798507462686567
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)
0.7798507462686567
metrics.confusion_matrix(y_test,y_pred)
array([[138,  21],
       [ 38,  71]], dtype=int64)
metrics.accuracy_score(y_test,y_pred)
0.7798507462686567
len(x_test)
268
EVALUTED THE MODEL PERFORMANCE USING
THE CLASSIFICATION REPORT
print(classification_report(y_test,y_pred))
              precision    recall  f1-score   support

           0       0.78      0.87      0.82       159
           1       0.77      0.65      0.71       109

    accuracy                           0.78       268
   macro avg       0.78      0.76      0.77       268
weighted avg       0.78      0.78      0.78       268

LogReg.coef_
array([[-1.71810035e-04, -8.62684880e-01, -3.06636588e+00,
        -1.35908169e-02, -1.63229448e-01, -3.53691697e-01,
         2.56935997e-03, -1.07265637e-01]])
LogReg.predict_proba(x_test)
array([[0.29263577, 0.70736423],
       [0.24350575, 0.75649425],
       [0.90723124, 0.09276876],
       [0.19729887, 0.80270113],
       [0.8270812 , 0.1729188 ],
       [0.47971371, 0.52028629],
       [0.92936702, 0.07063298],
       [0.83116965, 0.16883035],
       [0.05605183, 0.94394817],
       [0.48150965, 0.51849035],
       [0.05394834, 0.94605166],
       [0.89377416, 0.10622584],
       [0.82818174, 0.17181826],
       [0.35837144, 0.64162856],
       [0.40860175, 0.59139825],
       [0.87603975, 0.12396025],
       [0.06579236, 0.93420764],
       [0.64080523, 0.35919477],
       [0.82496275, 0.17503725],
       [0.89350854, 0.10649146],
       [0.90534265, 0.09465735],
       [0.04674774, 0.95325226],
       [0.89051301, 0.10948699],
       [0.28356244, 0.71643756],
       [0.89889734, 0.10110266],
       [0.90397481, 0.09602519],
       [0.25077733, 0.74922267],
       [0.76631004, 0.23368996],
       [0.91006821, 0.08993179],
       [0.87574314, 0.12425686],
       [0.98277694, 0.01722306],
       [0.1650527 , 0.8349473 ],
       [0.61191557, 0.38808443],
       [0.79866815, 0.20133185],
       [0.06816683, 0.93183317],
       [0.91773033, 0.08226967],
       [0.21290171, 0.78709829],
       [0.06068597, 0.93931403],
       [0.22671231, 0.77328769],
       [0.89784317, 0.10215683],
       [0.31445587, 0.68554413],
       [0.04463766, 0.95536234],
       [0.83669027, 0.16330973],
       [0.94054623, 0.05945377],
       [0.98265437, 0.01734563],
       [0.24580414, 0.75419586],
       [0.88985929, 0.11014071],
       [0.0453925 , 0.9546075 ],
       [0.83660659, 0.16339341],
       [0.08007458, 0.91992542],
       [0.88404731, 0.11595269],
       [0.79806841, 0.20193159],
       [0.9371911 , 0.0628089 ],
       [0.90696139, 0.09303861],
       [0.92863968, 0.07136032],
       [0.88910854, 0.11089146],
       [0.91012362, 0.08987638],
       [0.90374551, 0.09625449],
       [0.86976887, 0.13023113],
       [0.72700652, 0.27299348],
       [0.80840391, 0.19159609],
       [0.82597981, 0.17402019],
       [0.89787886, 0.10212114],
       [0.89858511, 0.10141489],
       [0.79195606, 0.20804394],
       [0.27133379, 0.72866621],
       [0.89722836, 0.10277164],
       [0.21876481, 0.78123519],
       [0.16094074, 0.83905926],
       [0.91516968, 0.08483032],
       [0.69955214, 0.30044786],
       [0.92159374, 0.07840626],
       [0.90958337, 0.09041663],
       [0.27426167, 0.72573833],
       [0.90762186, 0.09237814],
       [0.83460504, 0.16539496],
       [0.63609737, 0.36390263],
       [0.88423952, 0.11576048],
       [0.91733896, 0.08266104],
       [0.88541371, 0.11458629],
       [0.79416628, 0.20583372],
       [0.8168072 , 0.1831928 ],
       [0.09943704, 0.90056296],
       [0.07783789, 0.92216211],
       [0.08719138, 0.91280862],
       [0.90469796, 0.09530204],
       [0.89827269, 0.10172731],
       [0.15213906, 0.84786094],
       [0.77923999, 0.22076001],
       [0.92450043, 0.07549957],
       [0.89555788, 0.10444212],
       [0.42963899, 0.57036101],
       [0.89772008, 0.10227992],
       [0.89307632, 0.10692368],
       [0.05261223, 0.94738777],
       [0.4060777 , 0.5939223 ],
       [0.23288662, 0.76711338],
       [0.0870781 , 0.9129219 ],
       [0.90474753, 0.09525247],
       [0.27831024, 0.72168976],
       [0.89296924, 0.10703076],
       [0.89852296, 0.10147704],
       [0.89739947, 0.10260053],
       [0.15971323, 0.84028677],
       [0.91038555, 0.08961445],
       [0.09420317, 0.90579683],
       [0.91333266, 0.08666734],
       [0.27201042, 0.72798958],
       [0.2867111 , 0.7132889 ],
       [0.68360397, 0.31639603],
       [0.90854445, 0.09145555],
       [0.71888456, 0.28111544],
       [0.57044206, 0.42955794],
       [0.89769174, 0.10230826],
       [0.40991756, 0.59008244],
       [0.90291686, 0.09708314],
       [0.94058451, 0.05941549],
       [0.93936318, 0.06063682],
       [0.89511754, 0.10488246],
       [0.4067133 , 0.5932867 ],
       [0.89992147, 0.10007853],
       [0.90101484, 0.09898516],
       [0.08954888, 0.91045112],
       [0.63801235, 0.36198765],
       [0.93372943, 0.06627057],
       [0.0844851 , 0.9155149 ],
       [0.91448372, 0.08551628],
       [0.89027188, 0.10972812],
       [0.06105226, 0.93894774],
       [0.89720453, 0.10279547],
       [0.59261619, 0.40738381],
       [0.69483604, 0.30516396],
       [0.73900069, 0.26099931],
       [0.32413071, 0.67586929],
       [0.89135418, 0.10864582],
       [0.70527205, 0.29472795],
       [0.89358857, 0.10641143],
       [0.74297631, 0.25702369],
       [0.20407901, 0.79592099],
       [0.88815179, 0.11184821],
       [0.23962997, 0.76037003],
       [0.28563698, 0.71436302],
       [0.80057185, 0.19942815],
       [0.81727386, 0.18272614],
       [0.89936238, 0.10063762],
       [0.87221193, 0.12778807],
       [0.26620051, 0.73379949],
       [0.26912488, 0.73087512],
       [0.78236927, 0.21763073],
       [0.32292259, 0.67707741],
       [0.30905058, 0.69094942],
       [0.91946961, 0.08053039],
       [0.75025243, 0.24974757],
       [0.18849467, 0.81150533],
       [0.87125324, 0.12874676],
       [0.27745581, 0.72254419],
       [0.90608886, 0.09391114],
       [0.29020416, 0.70979584],
       [0.93939975, 0.06060025],
       [0.91319429, 0.08680571],
       [0.32262482, 0.67737518],
       [0.29149617, 0.70850383],
       [0.79199211, 0.20800789],
       [0.90209898, 0.09790102],
       [0.08465373, 0.91534627],
       [0.88988417, 0.11011583],
       [0.27742137, 0.72257863],
       [0.63177451, 0.36822549],
       [0.79294509, 0.20705491],
       [0.18490917, 0.81509083],
       [0.76665004, 0.23334996],
       [0.13395139, 0.86604861],
       [0.62473601, 0.37526399],
       [0.17276514, 0.82723486],
       [0.25053332, 0.74946668],
       [0.24534666, 0.75465334],
       [0.90990296, 0.09009704],
       [0.95770893, 0.04229107],
       [0.32302339, 0.67697661],
       [0.89379121, 0.10620879],
       [0.96777791, 0.03222209],
       [0.89558606, 0.10441394],
       [0.06046987, 0.93953013],
       [0.76064619, 0.23935381],
       [0.92824982, 0.07175018],
       [0.92785909, 0.07214091],
       [0.68505296, 0.31494704],
       [0.9274744 , 0.0725256 ],
       [0.39130291, 0.60869709],
       [0.16702424, 0.83297576],
       [0.22197763, 0.77802237],
       [0.91078976, 0.08921024],
       [0.16486524, 0.83513476],
       [0.89347722, 0.10652278],
       [0.78640193, 0.21359807],
       [0.96355702, 0.03644298],
       [0.4369088 , 0.5630912 ],
       [0.90017058, 0.09982942],
       [0.32942947, 0.67057053],
       [0.56924105, 0.43075895],
       [0.05916571, 0.94083429],
       [0.90622584, 0.09377416],
       [0.71945525, 0.28054475],
       [0.64311557, 0.35688443],
       [0.87062049, 0.12937951],
       [0.93884826, 0.06115174],
       [0.8948595 , 0.1051405 ],
       [0.64410534, 0.35589466],
       [0.65075541, 0.34924459],
       [0.07216114, 0.92783886],
       [0.76633513, 0.23366487],
       [0.58963312, 0.41036688],
       [0.84564552, 0.15435448],
       [0.29345855, 0.70654145],
       [0.91577275, 0.08422725],
       [0.90365343, 0.09634657],
       [0.15228691, 0.84771309],
       [0.07119006, 0.92880994],
       [0.95440888, 0.04559112],
       [0.82035367, 0.17964633],
       [0.88195631, 0.11804369],
       [0.28851043, 0.71148957],
       [0.09439545, 0.90560455],
       [0.89169455, 0.10830545],
       [0.10477839, 0.89522161],
       [0.89839991, 0.10160009],
       [0.90869041, 0.09130959],
       [0.41612808, 0.58387192],
       [0.68452946, 0.31547054],
       [0.9389086 , 0.0610914 ],
       [0.08023681, 0.91976319],
       [0.90606166, 0.09393834],
       [0.85875194, 0.14124806],
       [0.09622191, 0.90377809],
       [0.79952881, 0.20047119],
       [0.91585685, 0.08414315],
       [0.81537243, 0.18462757],
       [0.41162362, 0.58837638],
       [0.90771851, 0.09228149],
       [0.36933753, 0.63066247],
       [0.89658331, 0.10341669],
       [0.57262155, 0.42737845],
       [0.88571816, 0.11428184],
       [0.81082196, 0.18917804],
       [0.92664136, 0.07335864],
       [0.88209506, 0.11790494],
       [0.56235741, 0.43764259],
       [0.80800896, 0.19199104],
       [0.89948912, 0.10051088],
       [0.90964071, 0.09035929],
       [0.60925896, 0.39074104],
       [0.88992009, 0.11007991],
       [0.70824527, 0.29175473],
       [0.62331671, 0.37668329],
       [0.30945732, 0.69054268],
       [0.81090497, 0.18909503],
       [0.89375836, 0.10624164],
       [0.90156193, 0.09843807],
       [0.91460417, 0.08539583],
       [0.91118791, 0.08881209],
       [0.05043155, 0.94956845],
       [0.66753237, 0.33246763],
       [0.92250914, 0.07749086],
       [0.24156261, 0.75843739],
       [0.04650181, 0.95349819],
       [0.89778317, 0.10221683],
       [0.30148633, 0.69851367],
       [0.79575739, 0.20424261]])
# DXTENTACION_69
