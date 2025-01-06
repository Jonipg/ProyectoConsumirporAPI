
#imports
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

#Consyruccione que realize el partcicionado 
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]  #boundary significa algo parecido a limite o borde

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

df = pd.read_csv("datasetss/datasets/FinalDataset/Phishing.csv")

df["URL_Type_obf_Type"].value_counts()

is_null = df.isna().any()
is_null[is_null]


is_inf = df.isin([np.inf, -np.inf]).any()
is_inf[is_inf]

plt.figure(figsize=(12, 6))
plt.scatter(df["domainUrlRatio"][df['URL_Type_obf_Type'] == "phishing"], df["domainlength"][df['URL_Type_obf_Type'] == "phishing"], c="r", marker=".")
plt.scatter(df["domainUrlRatio"][df['URL_Type_obf_Type'] == "benign"], df["domainlength"][df['URL_Type_obf_Type'] == "benign"], c="g", marker="x")
plt.xlabel("domainUrlRatio", fontsize=13)
plt.ylabel("domainlength", fontsize=13)
plt.show()


train_set, val_set, test_set = train_val_test_split(df)

X_train = train_set.drop("URL_Type_obf_Type", axis=1)
y_train = train_set["URL_Type_obf_Type"].copy()

X_val = val_set.drop("URL_Type_obf_Type", axis=1)
y_val = val_set["URL_Type_obf_Type"].copy()

X_test = test_set.drop("URL_Type_obf_Type", axis=1)
y_test = test_set["URL_Type_obf_Type"].copy()


X_train=X_train.drop("argPathRatio", axis=1)
X_val=X_val.drop("argPathRatio", axis=1)
X_test=X_test.drop("argPathRatio", axis=1)


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")


X_train_prep=imputer.fit_transform(X_train)
X_val_prep=imputer.fit_transform(X_val)
X_test_prep=imputer.fit_transform(X_test)

X_train_prep= pd.DataFrame(X_train_prep,columns=X_train.columns, index=y_train.index)
X_val_prep= pd.DataFrame(X_val_prep,columns=X_val.columns, index=y_val.index)
X_test_prep= pd.DataFrame(X_test_prep,columns=X_test.columns, index=y_test.index)

X_train_prep.head(10)

is_null=X_train_prep.isna().any()
is_null[is_null]


X_train_reduced = X_train_prep[["domainUrlRatio", "domainlength"]].copy()
X_val_reduced= X_val_prep[["domainUrlRatio", "domainlength"]].copy()


X_train_reduced

from sklearn.svm import SVC

svm_clf = SVC(kernel="linear", C=50)
svm_clf.fit(X_train_reduced, y_train)

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=3)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

plt.figure(figsize=(12, 6))
plt.plot(X_train_reduced.values[:, 0][y_train=="phishing"], X_train_reduced.values[:, 1][y_train=="phishing"], "g^")
plt.plot(X_train_reduced.values[:, 0][y_train=="benign"], X_train_reduced.values[:, 1][y_train=="benign"], "bs")
plot_svc_decision_boundary(svm_clf, 0, 1)
plt.title("$C = {}$".format(svm_clf.C), fontsize=12)
plt.axis([0, 1, -100, 200])
plt.xlabel("domainUrlRatio", fontsize=15)
plt.ylabel("domainlength", fontsize=15)
plt.show()


y_pred=svm_clf.predict(X_val_reduced)


print("F1 Score:", f1_score(y_pred, y_val, pos_label="phishing"))
#diferencia de los valores de prediccion con los valores reales

svm_clf_sc = Pipeline([
        ("scaler", RobustScaler()),
        ("linear_svc", SVC(kernel="linear", C=50)),
    ])

svm_clf_sc.fit(X_train_reduced, y_train)



y_pred=svm_clf_sc.predict(X_val_reduced)

print("F1 Score", f1_score(y_pred, y_val, pos_label="phishing"))


from sklearn.svm import SVC

svm_clf = SVC(kernel="linear", C=1)
svm_clf.fit(X_train_prep, y_train)

y_pred=svm_clf.predict(X_val_prep)


print("F1 Score:", f1_score(y_pred, y_val, pos_label='phishing'))

y_train_num=y_train.factorize()[0]
y_val_num=y_val.factorize()[0]

from sklearn.datasets import make_moons
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=20, loss="hinge", random_state=42, max_iter=100000))
    ])

polynomial_svm_clf.fit(X_train_reduced, y_train_num)

def plot_dataset(X,y):
    plt.plot(X[:,0] [y==1], X[:,1] [y==1],"g.")
    plt.plot(X[:,0] [y==0], X[:,1] [y==0] , "b.")



def plot_predictions(clf,axes):
    x0s= np.linspace(axes[0],axes[1],100)
    x1s= np.linspace(axes[2],axes[3],100)
    x0, x1= np.meshgrid(x0s,x1s)
    X=np.c_[x0.ravel(),x1.ravel()] #redimencionando el arreglo
    y_pred=clf.predict(X).reshape(x0.shape)
    y_decision=clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0,x1,y_pred,cmap=plt.cm.brg,alpha=0.1) #para ponerlo mas obscuro
  

fig, axes = plt.subplots(ncols=2, figsize=(15,5), sharey=True)
plt.sca(axes[0])
plot_dataset(X_train_reduced.values, y_train_num)
plot_predictions(polynomial_svm_clf, [0, 1, -100, 250])
plt.xlabel("domainUrlRatio", fontsize=11)
plt.ylabel("domainlength", fontsize=11)
plt.sca(axes[1])
plot_predictions(polynomial_svm_clf, [0, 1, -100, 250])
plt.xlabel("domainUrlRatio", fontsize=11)
plt.ylabel("domainlength", fontsize=11)
plt.show()

y_prep=polynomial_svm_clf.predict(X_val_reduced)

print("F1 score",f1_score(y_prep,y_val_num))

svm_clf=SVC(kernel="poly",degree=3,coef0=10,C=20)
svm_clf.fit(X_train_reduced,y_train_num)

fig, axes = plt.subplots(ncols=2, figsize=(15,5), sharey=True)
plt.sca(axes[0])
plot_dataset(X_train_reduced.values, y_train_num)
plot_predictions(svm_clf, [0, 1, -100, 250])
plt.xlabel("domainUrlRatio", fontsize=11)
plt.ylabel("domainlength", fontsize=11)
plt.sca(axes[1])
plot_predictions(svm_clf, [0, 1, -100, 250])
plt.xlabel("domainUrlRatio", fontsize=11)
plt.ylabel("domainlength", fontsize=11)
plt.show()
