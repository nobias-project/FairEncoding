# %%
import pandas as pd
import random
import pdb

random.seed(0)
from collections import defaultdict

pd.set_option("display.max_columns", None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import shap

plt.rcParams["figure.figsize"] = [10, 5]
plt.rcParams.update({"font.size": 22})
plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8

import warnings

warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy.stats import wasserstein_distance
from xgboost import XGBClassifier
from category_encoders.target_encoder import TargetEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.ordinal import OrdinalEncoder

from tqdm import tqdm
from category_encoders import OneHotEncoder
from fairtools.utils import (
    explain,
    fit_predict,
    metric_calculator,
    plot_rolling,
    scale_output,
    columnDropperTransformer,
)


# %%
# Download and Load data
df = pd.read_csv("data/compas-scores-raw.csv")
# Target modfication
df["Score"] = df["DecileScore"]
df.loc[df["DecileScore"] > 4, "Score"] = 1
df.loc[df["DecileScore"] <= 4, "Score"] = 0
# Categorical features cleaning
df.loc[df["Ethnic_Code_Text"] == "African-Am", "Ethnic_Code_Text"] = "African-American"
# Cols that are going to be dropped
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
    "AssessmentReason",
    "Language",
    "Scale_ID",
    "IsCompleted",
    "IsDeleted",
    # "AssessmentType",
    "DecileScore",
    "RecSupervisionLevelText",
    # "DisplayText",
    # "ScaleSet",
    # "LegalStatus",
    # "CustodyStatus",
    "RawScore",
    "ScoreText",
]
df = df.drop(columns=cols)
# Some encoding of other categorical feats
df["Sex_Code_Text"] = pd.get_dummies(df["Sex_Code_Text"], prefix="Sex")["Sex_Male"]
df["ScaleSet"] = pd.get_dummies(df["ScaleSet"])["Risk and Prescreen"]
df = df.join(pd.get_dummies(df["DisplayText"]))
df = df.join(pd.get_dummies(df["AssessmentType"]))

## Drop categories with few values
df = df[(df["Ethnic_Code_Text"] != "Arabic") & (df["Ethnic_Code_Text"] != "Oriental")]

df = df.rename(columns={"Sex_Code_Text": "Sex"})
df = df.rename(columns={"Ethnic_Code_Text": "Ethnic"})
# %%
# Intersectional features
df["EthnicMarital"] = df["Ethnic"] + df["MaritalStatus"]
# Only categories that appear XX times
# ca_features = ca_features.groupby("MARgroup").filter(lambda x: len(x) > 10)

# Encode other cat features
df = OrdinalEncoder(
    cols=[
        "Agency_Text",
        "LegalStatus",
        "CustodyStatus",
        "DisplayText",
        "AssessmentType",
    ]
).fit_transform(df)

# %%
## Splitting
X = df.drop(columns=["Ethnic", "MaritalStatus", "Score"])
y = df[["Score"]]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=42)


# %%
# Remove groups that have small statistical mass
def func(pct, allvals):
    absolute = int(np.round(pct / 100.0 * np.sum(allvals)))
    return "{:.1f}%\n({:d})".format(pct, absolute)


# Auxiliary data for plottign
aux = pd.DataFrame(df["EthnicMarital"].value_counts())
colors = sns.color_palette("pastel")[0 : aux.shape[0]]
# create pie chart
plt.figure()
explode = (0.05,) * aux.shape[0]
plt.pie(
    aux.EthnicMarital.values,
    labels=aux.index,
    autopct=lambda pct: func(pct, aux.EthnicMarital.values),
    shadow=True,
    explode=explode,
)
plt.show()

# Auxiliary data for plottign # Intersect
aux = pd.DataFrame(df["Ethnic"].value_counts())
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
plt.show()


# %%
# Auxiliary functions
def fit_predict(modelo, enc, data, target, test):
    pipe = Pipeline([("encoder", enc), ("model", modelo)])
    pipe.fit(data, target)
    return pipe.predict(test)


def auc_group(model, data, y_true, dicc, group: str = "", min_samples: int = 10):
    aux = data.copy()
    aux["target"] = y_true
    cats = aux[group].value_counts()
    cats = cats[cats > min_samples].index.tolist()
    cats = cats + ["all"]

    if len(dicc) == 0:
        dicc = defaultdict(list, {k: [] for k in cats})

    for cat in cats:
        if cat != "all":
            aux2 = aux[aux[group] == cat]
            preds = model.predict_proba(aux2.drop(columns="target"))[:, 1]
            truth = aux2["target"]
            dicc[cat].append(roc_auc_score(truth, preds))
        elif cat == "all":
            dicc[cat].append(roc_auc_score(y_true, model.predict_proba(data)[:, 1]))
        else:
            pass

    return dicc


def explain(xgb: bool = True, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te):
    """
    Provide a SHAP explanation by fitting MEstimate and GBDT
    """
    if xgb:
        pipe = Pipeline([("encoder", MEstimateEncoder()), ("model", XGBClassifier())])
        pipe.fit(X_tr, y_tr)
        explainer = shap.Explainer(pipe[1])
        shap_values = explainer(pipe[:-1].transform(X_tr))
        shap.plots.beeswarm(shap_values)
        return pd.DataFrame(np.abs(shap_values.values), columns=X_tr.columns).sum()
    else:
        pipe = Pipeline(
            [("encoder", MEstimateEncoder()), ("model", LogisticRegression())]
        )
        pipe.fit(X_tr, y_tr)
        coefficients = pd.concat(
            [pd.DataFrame(X_tr.columns), pd.DataFrame(np.transpose(pipe[1].coef_))],
            axis=1,
        )
        coefficients.columns = ["feat", "val"]

        return coefficients.sort_values(by="val", ascending=False)


def calculate_cm(true, preds, metric="tpr"):
    # Obtain the confusion matrix
    cm = confusion_matrix(preds, true)

    #  https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    if metric == "tpr":
        return TPR[0]
    elif metric == "fpr":
        return FPR[0]
    else:
        raise ValueError("Metric not implemented")


def metric_calculator(
    modelo,
    data: pd.DataFrame,
    truth: pd.DataFrame,
    col: str,
    reference_group: str,
    compared_group: str = "All",
):
    """
    model: model to be used
    data: data to predict
    truth: ground truth labels
    col: column to be used as group
    reference_group: Reference group
    compared_group: Group to be compared, if all, all groups are compared and the sum is returned
    normalize: If True, the metric is normalized by ...
    """
    aux = data.copy()
    aux["target"] = truth

    if compared_group == "All":
        groups = data[col].unique()
        # Remove nans
        groups = groups[~pd.isnull(groups)]
    else:
        if compared_group not in data[col].unique():
            raise ValueError("Group not in data")
        groups = [compared_group]

    eof_sum = []
    dp_sum = []
    aao_sum = []

    for group2 in groups:
        # Filter the data
        g1 = data[data[col] == reference_group]
        g2 = data[data[col] == group2]

        # Filter the ground truth
        g1_true = aux[aux[col] == reference_group].target
        g2_true = aux[aux[col] == group2].target

        # Do predictions
        p1 = modelo.predict(g1)
        p2 = modelo.predict(g2)

        # Extract metrics for each group
        ## True Positive
        tpr1 = calculate_cm(p1, g1_true, metric="tpr")
        tpr2 = calculate_cm(p2, g2_true, metric="tpr")
        ## False Positive rates
        fpr1 = calculate_cm(p1, g1_true, metric="fpr")
        fpr2 = calculate_cm(p2, g2_true, metric="fpr")

        # Calculate fairness metrics
        ## Equal Opportunity Fairness
        eof = tpr1 - tpr2
        ## Demographic Parity
        dp = wasserstein_distance(p1, p2)
        ## Average Absolute Odds
        aao = (
            np.abs(tpr1 - fpr1) + np.abs(tpr2 - fpr2) - 1
        )  # The sum of the absolute differencesbetween the true positive rate and the false positive rates of the unprivileged group and thetrue positive rate and the false positive rates of the privileged group. For a fair model/data thismetric needs to be closer to zero
        eof_sum.append(eof)
        dp_sum.append(dp)
        aao_sum.append(aao)

    return (
        np.abs(eof_sum).max(),
        np.absolute(dp_sum).max(),
        np.absolute(aao_sum).max(),
    )


# %%
X["EthnicMarital"] = df["Ethnic"]
X["EthnicMarital"] = df["EthnicMarital"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=42)

# Train model
enc = columnDropperTransformer(columns="EthnicMarital")
m = Pipeline([("enc", enc), ("model", LogisticRegression())])
m.fit(X_tr, y_tr)
roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
# %%
m.named_steps["enc"].transform(X_tr).columns
# %%
# Intersectional Experiment
X = df.drop(columns=["Ethnic", "MaritalStatus", "Score"])
y = df[["Score"]]
encoder = [
    columnDropperTransformer(columns="EthnicMarital"),
    OneHotEncoder(),
    CatBoostEncoder(sigma=0.05),
    CatBoostEncoder(sigma=0.99),
    MEstimateEncoder(m=100_000),
]
dfs = []
for enc in encoder:
    for el in [True, False, "Other"]:
        intersect = el

        if intersect == True:
            X["EthnicMarital"] = df["EthnicMarital"]
        elif intersect == False:
            X["EthnicMarital"] = df["Ethnic"]
        else:
            X["EthnicMarital"] = df["MaritalStatus"]

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=42)

        # Train model
        m = Pipeline([("enc", enc), ("model", LogisticRegression())])
        m.fit(X_tr, y_tr)
        roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])

        # Calculate metrics
        res = {}
        for cat, num in X["EthnicMarital"].value_counts().items():
            COL = "EthnicMarital"
            if intersect == True:
                REFERENCE_GROUP = "CaucasianSingle"
            elif intersect == False:
                REFERENCE_GROUP = "Caucasian"
            else:
                REFERENCE_GROUP = "Single"
            GROUP2 = cat

            res[cat] = [
                metric_calculator(
                    modelo=m,
                    data=X,
                    truth=y,
                    col=COL,
                    reference_group=REFERENCE_GROUP,
                    compared_group=GROUP2,
                ),
                num,
            ]

        # Clean the results
        res = pd.DataFrame(res).T
        res.columns = ["fairness", "items"]
        res["items"] = res["items"].astype(int)
        res["eof"] = res["fairness"].apply(lambda x: x[0])
        res["dp"] = res["fairness"].apply(lambda x: x[1])
        res["aao"] = res["fairness"].apply(lambda x: x[2])
        res = res.drop(columns="fairness")
        dfs.append(res)


# %%
## Plot the results
non_enc_int = dfs[0].eof.max()
non_enc = dfs[1].eof.max()
non_enc_mar = dfs[2].eof.max()

ohe_int = dfs[3].eof.max()
ohe_non = dfs[4].eof.max()
ohe_mar = dfs[5].eof.max()

te_int = dfs[6].eof.max()
te_non = dfs[7].eof.max()
te_mar = dfs[8].eof.max()


teg_int = dfs[9].eof.max()
teg_non = dfs[10].eof.max()
teg_mar = dfs[11].eof.max()


tes_int = dfs[12].eof.max()
tes_non = dfs[13].eof.max()
tes_mar = dfs[14].eof.max()

labels = [
    "One Hot Encoding",
    "Target Encoder(Unreg)",
    "Target Encoder(Gaussian)",
    "Target Encoder(Smoothing)",
]
non = np.round([ohe_non, te_non, teg_non, tes_non], decimals=2)
mar = np.round([ohe_mar, te_mar, teg_mar, tes_mar], decimals=2)
inter = np.round([ohe_int, te_int, teg_int, tes_int], decimals=2)


x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
ax.axhline(
    y=non_enc_int, linestyle="-", label="Non Encoding(Inter)", color="lightcoral"
)
ax.axhline(
    y=non_enc, linestyle="-", label="Non Encoder(Ethnic) ", color="cornflowerblue"
)
ax.axhline(y=non_enc_mar, linestyle="-", label="Non Encoder(Marital)", color="k")
rects1 = ax.bar(
    x,
    mar,
    width,
    label="Marital",
    color="k",
)
rects2 = ax.bar(
    x + width,
    non,
    width,
    label="Ethnic",
    color="cornflowerblue",
    hatch=r"//",
)
rects3 = ax.bar(
    x + 2 * width,
    inter,
    width,
    label="Intersectional",
    color="lightcoral",
)


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Equal Opportunity Fairness Metric", fontsize=20)
ax.set_title("Max Fairness violation when using intersectional groups", fontsize=20)
ax.set_xticks(x, labels, fontsize=20)
ax.legend()

ax.bar_label(rects1, padding=3, fontsize=20)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

fig.tight_layout()
fig.savefig("images/compasInter.pdf", bbox_inches="tight")
plt.show()

# %%
