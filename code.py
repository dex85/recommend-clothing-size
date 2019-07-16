# --------------
# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Code starts here
# Load dataset
df = pd.read_json(path, lines=True)

df.columns = [x.replace(" ","_") for x in df.columns]

missing_data = df.count()

df.drop(['waist', 'bust', 'user_name','review_text','review_summary','shoe_size','shoe_width'], axis = 1, inplace = True)

X = df.drop(["fit"], axis = 1)
y = df["fit"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 6)
# Code ends here


# --------------
def plot_barh(df,col, cmap = None, stacked=False, norm = None):
    df.plot(kind='barh', colormap=cmap, stacked=stacked)
    fig = plt.gcf()
    fig.set_size_inches(24,12)
    plt.title("Category vs {}-feedback -  cloth {}".format(col, '(Normalized)' if norm else ''), fontsize= 20)
    plt.ylabel('Category', fontsize = 18)
    plot = plt.xlabel('Frequency', fontsize=18)


# Code starts here
g_by_category = df.groupby(['category'])
cat_fit = g_by_category['fit'].value_counts()
cat_fit = cat_fit.unstack()

cat_fit.plot.bar()
# Code ends here


# --------------
# Code starts here
cat_len = g_by_category['length'].value_counts()
cat_len = cat_fit.unstack()

plot_barh(cat_len, 'length')

# Code ends here


# --------------
# Code starts here
def get_cms(x):
    if(isinstance(x, float)):
        return 0
    else:
        if(len(x) < 4):
            return (int(x[0])*30.48)
        else:
            return (int(x[0])*30.48) + (int(x[4:-2])*2.54)

#print(X_train['height'][9])
X_train['height'] = X_train['height'].apply(get_cms)
X_test.height = X_test['height'].apply(get_cms)
#print(X_train.height[:2])
# Code ends here


# --------------
# Code starts here
missing_values_train = X_train.isna()
missing_values_test = X_test.isna()

X_train[['height', 'length', 'quality']].dropna(inplace = True)
X_test[['height', 'length', 'quality']].dropna(inplace = True)

#print(missing_values_train['height'] | missing_values_train['length'] | missing_values_train['quality'])

y_train = y_train[~(missing_values_train['height'] | missing_values_train['length'] | missing_values_train['quality'])]

y_test = y_test[~(missing_values_test['height'] | missing_values_test['length'] | missing_values_test['quality'])]
#y_test = y_test[~missing_values_test[['height', 'length', 'quality']]]

bra_mean_train = X_train['bra_size'].mean()
bra_mean_test = X_test['bra_size'].mean()

hips_mean_train = X_train['hips'].mean()
hips_mean_test = X_test['hips'].mean()

X_train['bra_size'].fillna(bra_mean_train, inplace = True)
X_test['bra_size'].fillna(bra_mean_test, inplace = True)

X_train['hips'].fillna(hips_mean_train, inplace = True)
X_test['hips'].fillna(hips_mean_test, inplace = True)

mode_1 = X_train['cup_size'].mode()[0]
mode_2 = X_test['cup_size'].mode()[0]

X_train['cup_size'].fillna(mode_1, inplace = True)
X_test['cup_size'].fillna(mode_2, inplace = True)
# Code ends here


# --------------
# Code starts here




X_train = pd.get_dummies(data=X_train, columns = ['category','cup_size','length'])
X_test = pd.get_dummies(data=X_test, columns = ['category','cup_size','length'])

# Code ends here


# --------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


# Code starts here
model = DecisionTreeClassifier(random_state = 6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

score = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average = None)

# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# parameters for grid search
parameters = {'max_depth':[5,10],'criterion':['gini','entropy'],'min_samples_leaf':[0.5,1]}

# Code starts here
model = DecisionTreeClassifier(random_state = 6)
grid = GridSearchCV(estimator = model, param_grid = parameters)

grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
# Code ends here


