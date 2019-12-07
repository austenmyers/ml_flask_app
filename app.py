import pandas
import numpy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
url = "Dataset/adult.csv"
df = pandas.read_csv(url)

print("df cols: ")
print(df.columns)

# filling missing values
col_names = [
"age", #1
"workclass", #2
"fnlwgt", #3
"education", #4
"education-num", #5
"marital-status", #6
"occupation", #7
"relationship", #8
"race", #9
"gender", #10
"capital-gain", #11
"capital-loss", #12
"hours-per-week", #13
"native-country", #14
"income"
]

df.columns = col_names

print("df cols with update: ")
print(df.columns)

for c in df.columns:
    df[c] = df[c].replace("?", numpy.NaN)

df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

#discretisation
df.replace(['Divorced', 'Married-AF-spouse',
              'Married-civ-spouse', 'Married-spouse-absent',
              'Never-married','Separated','Widowed'],
             ['divorced','married','married','married',
              'not married','not married','not married'], inplace = True)

#label Encoder
category_col =['workclass', 'race', 'education','marital-status', 'occupation',
               'relationship', 'gender', 'native-country', 'income']
labelEncoder = preprocessing.LabelEncoder()

print("df cols: ")
print(df.columns)

# creating a map of all the numerical values of each categorical labels.
mapping_dict={}
for col in category_col:
    df[col] = labelEncoder.fit_transform(df[col])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col]=le_name_mapping
print(mapping_dict)

#droping redundant columns
df=df.drop(['fnlwgt','education-num'], axis=1)

print("df head: ")
print(df.head())

# split into train cols and label
X = df.values[:, 0:12]
Y = df.values[:,12]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
dt_clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train, y_train)
y_pred_gini = dt_clf_gini.predict(X_test)

print ("Desicion Tree using Gini Index\nAccuracy is ", accuracy_score(y_test,y_pred_gini)*100 )

#creating and training a model
#serializing our model to a file called model.pkl
filename = 'model'
outfile = open(filename,'wb')
pickle.dump(dt_clf_gini, outfile)
outfile.close()
