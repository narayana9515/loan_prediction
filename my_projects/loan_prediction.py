import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import warnings
d1=pd.read_csv(r"C:\\Users\\NARAYANA\\OneDrive\\Documents\\datasets\\loan_data.csv")
d2=pd.DataFrame(d1)
d=d2.head(15)
d.isnull().sum()                       #for count of null values in each row
d.isnull().sum()/d.shape[0]*100                              #for calculating percentage of null values in each row
d=d.drop(["person_gender","person_emp_exp","person_home_ownership","person_education","cb_person_cred_hist_length"],axis=1)
d=pd.DataFrame(d)
d
sns.displot(d['person_age'])     #displot used for single column representation
plt.grid(axis='y') 
sns.displot(d['loan_intent'])     #displot used for single column representation
plt.grid(axis='y')
sns.displot(d['previous_loan_defaults_on_file'])     #displot used for single column representation
plt.grid(axis='y') 
sns.jointplot(x='person_age',y='person_income',data=d)       #for multiple column representation
plt.grid(axis='both')   
X=d.iloc[:,:8]
Y=d.iloc[:,8]
print(pd.DataFrame(X))
print("___________________________________________________________________")
print(pd.DataFrame(Y))     
from sklearn.preprocessing import LabelEncoder
X=pd.get_dummies(X,prefix=['loan_intent'],columns=['loan_intent'],dtype=int,sparse=False)
le=LabelEncoder()
X["previous_loan_defaults_on_file"]=le.fit_transform(X["previous_loan_defaults_on_file"])
print(X)
print("___________________________________________________________________")
print(Y)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X[["person_income","loan_amnt","credit_score"]]=sc.fit_transform(X[["person_income","loan_amnt","credit_score"]])
print(X)
print("___________________________________________________________________")
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
print(X_train)                                                                      #splitting of data
print("_________________________________________________________________")
print(X_test)
print("_________________________________________________________________")
print(Y_train)
print("_________________________________________________________________")
print(Y_test)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
c_knn= KNeighborsClassifier(n_neighbors=2)
c_knn.fit(X_train,Y_train)
Y_pred=c_knn.predict(X_test)
accu=metrics.accuracy_score(Y_test,Y_pred)
print(f"accuracy={accu}")
age= 24   
income= -0.098919
loan_amnt=0.705586 
loan_in_rate=14.27 
l_per_income=0.53 
credit_score= -0.347742  
p_l_d_o_f=0

age1= 24      
income1= -0.098919
loan_amnt1=0.705586 
loan_in_rate1=14.27 
l_per_income1=0.53 
credit_score1= -0.347742  
p_l_d_o_f1=0

sample=[[age,income,loan_amnt,0,0,0,0,1,loan_in_rate,l_per_income,credit_score,p_l_d_o_f],
        [age1,income1,loan_amnt1,0,0,0,0,1,loan_in_rate1,l_per_income1,credit_score1,p_l_d_o_f1]]
preds=c_knn.predict(sample)
predics=[d.loan_status[p] for p in preds] 
print(f"predictions={predics}")
for i in predics:
    if i == 1:
        print("Loan Approved")
    else:
        print("Loan Denied")




       
