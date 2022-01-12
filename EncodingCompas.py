# %%
import pandas as pd

pd.set_option("display.max_columns", None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

from matplotlib import rcParams

plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8

# from pandas_profiling import ProfileReportofileReport

import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [10, 5]
import shap

from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBRegressor, XGBClassifier
import shap

from category_encoders.target_encoder import TargetEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.james_stein import JamesSteinEncoder

from tqdm import tqdm
from collections import defaultdict
from category_encoders import OneHotEncoder
from fairtools.utils import (
    explain,
    auc_group,
    fit_predict,
    metric_calculator,
    plot_rolling,
    scale_output,
)

# %%
df = pd.read_csv("data/compas-scores-raw.csv")

df["Score"] = df["DecileScore"]

df.loc[df["DecileScore"] > 4, "Score"] = 1
df.loc[df["DecileScore"] <= 4, "Score"] = 0

df.loc[df["Ethnic_Code_Text"] == "African-Am", "Ethnic_Code_Text"] = "African-American"

cols = [
    "Person_ID",
    "AssessmentID",
    "Case_ID",
    "LastName",
    "FirstName",
    "MiddleName",
    "DateOfBirth",
    "ScaleSet_ID",
    "Screening_Date",
    "RecSupervisionLevel",
    # "Agency_Text",
    # "AssessmentReason",
    "Language",
    "Scale_ID",
    # "IsCompleted",
    # "IsDeleted",
    # "AssessmentType",
    "DecileScore",
    "RecSupervisionLevelText",
    # "DisplayText",
    # "ScaleSet",
    # "LegalStatus",
    # "CustodyStatus",
]


df = df.drop(columns=cols)

possible_targets = ["RawScore", "ScoreText", "Score"]

X = df.drop(columns=possible_targets)
y = df[["Score"]]
X["Sex_Code_Text"] = pd.get_dummies(X["Sex_Code_Text"], prefix="Sex")["Sex_Male"]
X["ScaleSet"] = pd.get_dummies(X["ScaleSet"])["Risk and Prescreen"]

X.columns = [
    "Agency_Text",
    "Sex",
    "Ethnic",
    "ScaleSet",
    "AssessmentReason",
    "LegalStatus",
    "CustodyStatus",
    "MaritalStatus",
    "DisplayText",
    "AssessmentType",
    "IsCompleted",
    "IsDeleted",
]


# %%
# Remove groups that have small statistical mass
filter_value = 323

aux = pd.DataFrame(X["Ethnic"].value_counts())
aux2 = pd.DataFrame(
    data={"Ethnic": aux[aux.Ethnic < filter_value].sum()[0]}, index=["Minor Groups"]
)
aux = aux.append(aux2)
aux = aux[aux.Ethnic >= filter_value]


def func(pct, allvals):
    absolute = int(np.round(pct / 100.0 * np.sum(allvals)))
    return "{:.1f}%\n({:d})".format(pct, absolute)


colors = sns.color_palette("pastel")[0 : aux.shape[0]]


# create pie chart
plt.figure()
explode = (0.05,) * aux.shape[0]

plt.pie(
    aux.Ethnic.values,
    labels=aux.index,
    autopct=lambda pct: func(pct, aux.Ethnic.values),
    shadow=True,
    explode=explode,
)
plt.savefig("images/pieCompas.png")
plt.savefig("images/pieCompas.eps", format="eps")


# %%
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=42)
# %%
def fair_encoder(model, param: list, enc: str = "mestimate", un_regularize: list = []):
    auc = {}
    metrica = []
    auc_tot = []

    allowed_enc = [
        "mestimate",
        "targetenc",
        "leaveoneout",
        "ohe",
        "woe",
        "james",
        "catboost",
    ]
    assert (
        enc in allowed_enc
    ), "Encoder not available or check for spelling mistakes: {}".format(allowed_enc)

    cols_enc = set(X_tr.columns) - set(un_regularize)

    for m in tqdm(param):
        if enc == "mestimate":
            encoder = Pipeline(
                [
                    ("reg", MEstimateEncoder(m=m, cols=cols_enc)),
                    ("unreg", MEstimateEncoder(m=0, cols=un_regularize)),
                ]
            )
        elif enc == "targetenc":
            encoder = TargetEncoder(smoothing=m)
        elif enc == "leaveoneout":
            encoder = Pipeline(
                [
                    ("reg", LeaveOneOutEncoder(sigma=m, cols=cols_enc)),
                    ("unreg", LeaveOneOutEncoder(sigma=0, cols=un_regularize)),
                ]
            )
        elif enc == "ohe":
            encoder = OneHotEncoder(handle_missing=-1)
        elif enc == "woe":
            encoder = WOEEncoder(randomized=True, sigma=m)
        elif enc == "james":
            encoder = JamesSteinEncoder(randomized=True, sigma=m)
        elif enc == "catboost":
            encoder = Pipeline(
                [
                    ("reg", CatBoostEncoder(a=1, sigma=m, cols=cols_enc)),
                    ("unreg", CatBoostEncoder(a=1, sigma=0, cols=un_regularize)),
                ]
            )

        pipe = Pipeline([("encoder", encoder), ("model", model)])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)

        metrica.append(
            metric_calculator(
                modelo=pipe,
                data=X_tr,
                truth=y_tr,
                col=COL,
                group1=GROUP1,
                group2=GROUP2,
            )
        )
        auc = auc_group(model=pipe, data=X_te, y_true=y_te, dicc=auc, group=COL)
        auc_tot.append(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]))

    # Results formatting
    res = pd.DataFrame(index=param)
    res["fairness_metric"] = metrica
    res["auc_tot"] = auc_tot
    res["auc_" + GROUP1] = auc[GROUP1]
    res["auc_" + GROUP2] = auc[GROUP2]

    return res


# %%
# Experiment parameters
COL = "Ethnic"
GROUP1 = "African-American"
GROUP2 = "Caucasian"

# %%
cols_enc = [
    "Agency_Text",
    "Sex",
    "ScaleSet",
    "AssessmentReason",
    "LegalStatus",
    "CustodyStatus",
    "DisplayText",
    "AssessmentType",
    "IsCompleted",
    "IsDeleted",
]
# %%
## LR
one_hot1 = fair_encoder(model=LogisticRegression(), enc="ohe", param=[0])

PARAM = np.linspace(0, 1, 100)
gaus1 = fair_encoder(
    model=LogisticRegression(), enc="catboost", param=PARAM, un_regularize=cols_enc
)
PARAM = np.linspace(0, 10_000, 100)
smooth1 = fair_encoder(
    model=LogisticRegression(), enc="mestimate", param=PARAM, un_regularize=cols_enc
)


# %%
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

# LR
axs[0].set_title("Logistic Regression + Gaussian Noise")
axs[0].scatter(
    gaus1["auc_tot"].values,
    gaus1["fairness_metric"].values,
    s=100,
    c=gaus1.index.values,
    cmap="Reds",
    label="Regularization Parameter (Darker=High)",
)
axs[0].scatter(
    y=one_hot1.fairness_metric, x=one_hot1.auc_tot, s=100, label="One Hot Encoder"
)
axs[0].legend()

axs[0].set(xlabel="AUC")
axs[1].set(xlabel="AUC")
axs[0].set(ylabel="Equal opportunity fairness (TPR)")


axs[1].set_title("Logistic Regression + Smoothing Regularizer")
axs[1].scatter(
    smooth1["auc_tot"].values,
    smooth1["fairness_metric"].values,
    s=100,
    c=smooth1.index.values,
    cmap="Reds",
    label="Regularization Parameter (Darker=High)",
)
axs[1].scatter(
    y=one_hot1.fairness_metric, x=one_hot1.auc_tot, s=100, label="One Hot Encoder"
)
fig.savefig("images/compasLinear.png")
fig.show()
# %%
fig, axs = plt.subplots(1, 2, sharex=True)

fig.suptitle("Gaussian regularization target encoding")
aux = (
    gaus1[["auc_tot", "auc_African-American", "auc_Caucasian"]]
    .rolling(5)
    .mean()
    .dropna()
)


for col in aux.columns:
    axs[0].plot(aux[col], label=col)
    # plt.fill_between(aux.index,(aux[col] - stand[col]),(aux[col] + stand[col]),# color="b",alpha=0.1,)
axs[0].legend()
axs[0].set_title("Model performance")
axs[0].set_ylabel("AUC")
axs[0].set_xlabel("Regularization parameter")

aux = gaus1[["fairness_metric"]].rolling(5).mean().dropna()

axs[1].plot(aux["fairness_metric"], label=GROUP1 + " vs " + GROUP2, color="r")

axs[1].legend()
axs[1].set_title("Fairness Metric")
axs[1].set_ylabel("Equal opportunity fairness (TPR)")
axs[1].set_xlabel("Regularization parameter")
plt.savefig("images/compassHyperGaussian.png")
plt.show()
# %%
