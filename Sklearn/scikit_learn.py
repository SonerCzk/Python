import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

sales_data = pd.read_csv("WA_Fn-UseC_-Sales-Win-Loss.csv").sample(1000)

register_count=sales_data["Region"].value_counts()

fig,axes=plt.subplots(1,2)
axes[0].bar(register_count.index,register_count)
axes[1].pie(register_count,labels=register_count.index,autopct="%.2f")
plt.show()

print(len(sales_data))

le=LabelEncoder()

sales_data["Supplies Subgroup"]=le.fit_transform(sales_data["Supplies Subgroup"])
sales_data["Supplies Group"]=le.fit_transform(sales_data["Supplies Group"])
sales_data["Region"]=le.fit_transform(sales_data["Region"])
sales_data["Route To Market"]=le.fit_transform(sales_data["Route To Market"])
sales_data["Opportunity Result"]=le.fit_transform(sales_data["Opportunity Result"])
sales_data["Competitor Type"]=le.fit_transform(sales_data["Competitor Type"])

cols=[col for col in sales_data.columns if col not in ["Opportunity Number","Opportunity Result"]]

x=sales_data[cols]
y=sales_data["Opportunity Result"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
dt_pred=dtc.predict(x_test)
print("DT Accuracy:",accuracy_score(y_test,dt_pred))

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
print("KNN Accuracy:",accuracy_score(y_test,knn.predict(x_test)))

svc=SVC()
svc.fit(x_train,y_train)
print("SVC Accuracy:",accuracy_score(y_test,svc.predict(x_test)))

print("train size: ",x_train.shape[0])
print("test size:",len(x_test))


