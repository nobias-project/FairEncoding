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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import wasserstein_distance
from xgboost import XGBClassifier
from category_encoders.target_encoder import TargetEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.james_stein import JamesSteinEncoder

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

from folktables import (
    ACSDataSource,
    ACSIncome,
    ACSEmployment,
    ACSMobility,
    ACSPublicCoverage,
    ACSTravelTime,
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
# Split data
X = df.drop(columns="Score")

y = df[["Score"]]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=42)
# %%
# Auxiliary data for plotting
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
        aao = np.abs(tpr1 - fpr1) + np.abs(
            tpr2 - fpr2
        )  # The sum of the absolute differencesbetween the true positive rate and the false positive rates of the unprivileged group and thetrue positive rate and the false positive rates of the privileged group. For a fair model/data thismetric needs to be closer to zero
        eof_sum.append(eof)
        dp_sum.append(dp)
        aao_sum.append(aao)

    return (
        np.abs(eof_sum).sum(),
        np.absolute(dp_sum).sum(),
        np.absolute(aao_sum).sum(),
    )


# %%
# Train model
m = Pipeline([("enc", CatBoostEncoder(sigma=0.5)), ("model", LogisticRegression())])
m.fit(X_tr, y_tr)
roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
# %%
res = {}
for cat, num in X["Ethnic"].value_counts().items():
    COL = "Ethnic"
    REFERENCE_GROUP = "Asian"
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
res
# %%
def plot_rolling(data, roll_mean: int = 5, roll_std: int = 20):

    aux = data.rolling(roll_mean).mean().dropna()
    stand = data.rolling(roll_std).quantile(0.05, interpolation="lower").dropna()
    plt.figure()
    for col in data.columns:
        plt.plot(aux[col], label=col)
        # plt.fill_between(aux.index,(aux[col] - stand[col]),(aux[col] + stand[col]),# color="b",alpha=0.1,)
    plt.legend()
    plt.show()


def scale_output(data):
    return pd.DataFrame(
        StandardScaler().fit_transform(data), columns=data.columns, index=data.index
    )


# %%
# Experiment
def fair_encoder(model, param: list, enc: str = "mestimate", drop_cols: list = []):
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
        "drop",
    ]
    assert (
        enc in allowed_enc
    ), "Encoder not available or check for spelling mistakes: {}".format(allowed_enc)

    cols_enc = set(X_tr.columns) - set(drop_cols)
    cols_enc = X_tr.select_dtypes(include=["object", "category"]).columns

    for m in tqdm(param):
        if enc == "mestimate":
            encoder = MEstimateEncoder(m=m, cols=cols_enc)
        elif enc == "targetenc":
            encoder = TargetEncoder(smoothing=m)
        elif enc == "leaveoneout":
            encoder = LeaveOneOutEncoder(sigma=m, cols=cols_enc)
        elif enc == "ohe":
            encoder = OneHotEncoder(handle_missing=-1)
        elif enc == "woe":
            encoder = WOEEncoder(randomized=True, sigma=m)
        elif enc == "james":
            encoder = JamesSteinEncoder(randomized=True, sigma=m)
        elif enc == "catboost":
            encoder = CatBoostEncoder(a=1, sigma=m, cols=cols_enc)
        elif enc == "drop":
            encoder = Pipeline(
                [
                    ("drop", columnDropperTransformer(columns=cols_enc)),
                ]
            )

        pipe = Pipeline([("encoder", encoder), ("model", model)])
        pipe.fit(X_tr, y_tr)
        metrica.append(
            metric_calculator(
                modelo=pipe,
                data=X_tr,
                truth=y_tr,
                col=COL,
                reference_group=GROUP1,
                compared_group=GROUP2,
            )
        )
        auc = auc_group(model=pipe, data=X_te, y_true=y_te, dicc=auc, group=COL)
        auc_tot.append(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]))

    # Results formatting
    res = pd.DataFrame(index=param)
    res["fairness_metric"] = metrica
    ## Decompress fairness metrics
    res["eof"] = res["fairness_metric"].apply(lambda x: x[0])
    res["dp"] = res["fairness_metric"].apply(lambda x: x[1])
    res["aao"] = res["fairness_metric"].apply(lambda x: x[2])
    res = res.drop(columns="fairness_metric")

    ## AUC
    auc = pd.DataFrame(auc, index=param)
    res["auc_tot"] = auc_tot  # Macro
    res["auc_micro"] = auc.drop(columns=["all"]).mean(axis=1)
    res["auc_" + GROUP1] = auc[GROUP1]
    for col1 in auc.columns:
        try:
            res["auc_" + col1] = auc[col1]
        except:
            print("Eventually should be fixed", col1)
    return res


# %%
# Experiment parameters
COL = "Ethnic"
GROUP1 = "Caucasian"
GROUP2 = "All"
# Lenght of the linspace
POINTS = 50
# %%
## LR Experiment
no_encoding1 = fair_encoder(
    model=LogisticRegression(), enc="drop", drop_cols=COL, param=[0]
)
one_hot1 = fair_encoder(model=LogisticRegression(), enc="ohe", param=[0])

PARAM = np.linspace(0, 1, POINTS)
gaus1 = fair_encoder(
    model=LogisticRegression(),
    enc="catboost",
    param=PARAM,
)
PARAM = np.linspace(0, 100_000, POINTS)
smooth1 = fair_encoder(
    model=LogisticRegression(),
    enc="mestimate",
    param=PARAM,
)
# %%
# Visualize results
##################
### Figure 1 #####
##################
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
# LR
axs[0].set_title("Logistic Regression + Gaussian Noise")
### Fairness metrics plotting
axs[0].scatter(
    gaus1["auc_tot"].values,
    gaus1["eof"].values,
    s=100,
    c=gaus1.index.values,
    cmap="Reds",
    label="EOF Regularization Parameter (Darker=High)",
)
axs[0].scatter(
    gaus1["auc_tot"].values,
    gaus1["dp"].values,
    s=100,
    c=gaus1.index.values,
    cmap="Blues",
    label="Demographic Parity",
)
axs[0].scatter(
    gaus1["auc_tot"].values,
    gaus1["aao"].values,
    s=100,
    c=gaus1.index.values,
    cmap="Greens",
    label="Average Absolute Odds",
)
### ONE-HOT
axs[0].scatter(
    y=one_hot1["eof"],
    x=one_hot1.auc_tot,
    c="r",
    marker="x",
    s=100,
    label="One Hot Encoder EOF",
)
axs[0].scatter(
    y=one_hot1["dp"],
    x=one_hot1.auc_tot,
    c="b",
    marker="x",
    s=100,
    label="One Hot Encoder DP",
)
axs[0].scatter(
    y=one_hot1["aao"],
    x=one_hot1.auc_tot,
    c="g",
    marker="x",
    s=100,
    label="One Hot Encoder AAO",
)

## No Encoding - Protected attribute is out
axs[0].scatter(
    y=no_encoding1["eof"],
    x=no_encoding1.auc_tot,
    c="r",
    marker="*",
    s=100,
    label="No encoding EOF",
)
axs[0].scatter(
    y=no_encoding1["dp"],
    x=no_encoding1.auc_tot,
    c="b",
    marker="*",
    s=100,
    label="No Encoding DP",
)
axs[0].scatter(
    y=no_encoding1["aao"],
    x=no_encoding1.auc_tot,
    c="g",
    marker="*",
    s=100,
    label="No Encoding AAO",
)

### Figure labels
axs[0].legend()
axs[0].set(xlabel="AUC")
axs[1].set(xlabel="AUC")
axs[0].set(ylabel="Fairness metrics")
axs[1].set_title("Logistic Regression + Smoothing Regularizer")
leg = axs[0].get_legend()
leg.legendHandles[0].set_color("red")
leg.legendHandles[1].set_color("blue")
leg.legendHandles[2].set_color("green")

axs[1].scatter(
    smooth1["auc_tot"].values,
    smooth1["eof"].values,
    s=100,
    c=smooth1.index.values,
    cmap="Reds",
    label="EOF Regularization Parameter (Darker=High)",
)
axs[1].scatter(
    smooth1["auc_tot"].values,
    smooth1["dp"].values,
    s=100,
    c=smooth1.index.values,
    cmap="Blues",
    label="Demographic Parity",
)
axs[1].scatter(
    smooth1["auc_tot"].values,
    smooth1["aao"].values,
    s=100,
    c=smooth1.index.values,
    cmap="Greens",
    label="Average Absolute Odds",
)

### ONE-HOT
axs[1].scatter(
    y=one_hot1["eof"],
    x=one_hot1.auc_tot,
    c="r",
    marker="x",
    s=100,
    label="One Hot Encoder EOF",
)
axs[1].scatter(
    y=one_hot1["dp"],
    x=one_hot1.auc_tot,
    c="b",
    marker="x",
    s=100,
    label="One Hot Encoder DP",
)
axs[1].scatter(
    y=one_hot1["aao"],
    x=one_hot1.auc_tot,
    c="g",
    marker="x",
    s=100,
    label="One Hot Encoder AAO",
)
## No Encoding - Protected attribute is out
axs[1].scatter(
    y=no_encoding1["eof"],
    x=no_encoding1.auc_tot,
    c="r",
    marker="*",
    s=100,
    label="No encoding EOF",
)
axs[1].scatter(
    y=no_encoding1["dp"],
    x=no_encoding1.auc_tot,
    c="b",
    marker="*",
    s=100,
    label="No Encoding DP",
)
axs[1].scatter(
    y=no_encoding1["aao"],
    x=no_encoding1.auc_tot,
    c="g",
    marker="*",
    s=100,
    label="No Encoding AAO",
)
fig.savefig("images/encTheory.pdf", bbox_inches="tight")
fig.show()
# %%
### Figure 2 #####
##################
"""
This figure shows the effect of the smoothing regularizer on the AUC of the model
"""
fig, axs = plt.subplots(1, 2, sharex=True)

fig.suptitle("Gaussian regularization target encoding")
aux = gaus1.drop(columns=["dp", "aao", "eof"])  # .rolling(5).mean().dropna()

for col in aux.columns:
    axs[0].plot(aux[col], label=col)
    # plt.fill_between(aux.index,(aux[col] - stand[col]),(aux[col] + stand[col]),# color="b",alpha=0.1,)
axs[0].legend()
axs[0].set_title("Model performance")
axs[0].set_ylabel("AUC")
axs[0].set_xlabel("Regularization parameter")

aux = gaus1[["eof", "dp", "aao"]]  # .rolling(5).mean().dropna()

axs[1].plot(aux["eof"], label="EOF " + GROUP1 + " vs " + GROUP2, color="r")
axs[1].plot(aux["dp"], label="DP " + GROUP1 + " vs " + GROUP2, color="b")
axs[1].plot(aux["aao"], label="AAO" + GROUP1 + " vs " + GROUP2, color="g")

axs[1].legend()
axs[1].set_title("Fairness Metric")
axs[1].set_ylabel("Fairness Metrics")
axs[1].set_xlabel("Regularization parameter")
plt.savefig("images/compassHyperGaussian.pdf", bbox_inches="tight")
plt.show()
### Figure 3 #####
##################
fig, axs = plt.subplots(1, 2, sharex=True)

fig.suptitle("Smoothing regularization target encoding")
aux = smooth1.drop(columns=["dp", "aao", "eof"])  # .rolling(5).mean().dropna()


for col in aux.columns:
    axs[0].plot(aux[col], label=col)
    # plt.fill_between(aux.index,(aux[col] - stand[col]),(aux[col] + stand[col]),# color="b",alpha=0.1,)
axs[0].legend()
axs[0].set_title("Model performance")
axs[0].set_ylabel("AUC")
axs[0].set_xlabel("Regularization parameter")

aux = smooth1[["dp", "eof", "aao"]]  # .rolling(5).mean().dropna()
axs[1].plot(aux["eof"], label="EOF " + GROUP1 + " vs " + GROUP2, color="r")
axs[1].plot(aux["dp"], label="DP " + GROUP1 + " vs " + GROUP2, color="b")
axs[1].plot(aux["aao"], label="AAO" + GROUP1 + " vs " + GROUP2, color="g")

axs[1].legend()
axs[1].set_title("Fairness Metrics")
axs[1].set_ylabel("")
axs[1].set_xlabel("Regularization parameter")
plt.savefig("images/compassHyperSmoothing.pdf", bbox_inches="tight")
plt.show()


# %%
### Figure 4 #####
##################
"""
3 Models are trained with different regularization parameters
"""
# TRAINING
## LR -- Removed since it should be already trained
# one_hot1 = fair_encoder(model=LogisticRegression(), enc="ohe", param=[0])

# PARAM = np.linspace(0, 1, 50)
# gaus1 = fair_encoder(model=LogisticRegression(), enc="catboost", param=PARAM)
# PARAM = np.linspace(0, 100, 50)
# smooth1 = fair_encoder(model=LogisticRegression(), enc="mestimate", param=PARAM)
## DT
one_hot2 = fair_encoder(model=MLPClassifier(), enc="ohe", param=[0])
no_encoding2 = fair_encoder(
    model=MLPClassifier(), enc="drop", drop_cols=COL, param=[0]
)
PARAM = np.linspace(0, 1, POINTS)
gaus2 = fair_encoder(
    model=MLPClassifier(),
    enc="catboost",
    param=PARAM,
)
PARAM = np.linspace(0, 100_000, POINTS)
smooth2 = fair_encoder(
    model=MLPClassifier(),
    enc="mestimate",
    param=PARAM,
)
## GBDT
one_hot3 = fair_encoder(model=XGBClassifier(), enc="ohe", param=[0])
no_encoding3 = fair_encoder(model=XGBClassifier(), enc="drop", drop_cols=COL, param=[0])
PARAM = np.linspace(0, 1, POINTS)
gaus3 = fair_encoder(
    model=XGBClassifier(),
    enc="catboost",
    param=PARAM,
)
PARAM = np.linspace(0, 100_000, POINTS)
smooth3 = fair_encoder(
    model=XGBClassifier(),
    enc="mestimate",
    param=PARAM,
)
# %%
## VIZ 3 MODELS
########################
########################

fig, axs = plt.subplots(3, 2, figsize=(15, 15), sharex=True, sharey=True)
######### LR #########
########################
axs[0, 0].set_title("Logistic Regression + Gaussian Noise")
### Fairness metrics plotting
axs[0, 0].scatter(
    gaus1["auc_tot"].values,
    gaus1["eof"].values,
    s=100,
    c=gaus1.index.values,
    cmap="Reds",
    label="EOF Regularization Parameter (Darker=High)",
)
axs[0, 0].scatter(
    gaus1["auc_tot"].values,
    gaus1["dp"].values,
    s=100,
    c=gaus1.index.values,
    cmap="Blues",
    label="Demographic Parity",
)
axs[0, 0].scatter(
    gaus1["auc_tot"].values,
    gaus1["aao"].values,
    s=100,
    c=gaus1.index.values,
    cmap="Greens",
    label="Average Absolute Odds",
)
### ONE-HOT
axs[0, 0].scatter(
    y=one_hot1["eof"],
    x=one_hot1.auc_tot,
    c="r",
    marker="x",
    s=100,
    label="One Hot Encoder EOF",
)
axs[0, 0].scatter(
    y=one_hot1["dp"],
    x=one_hot1.auc_tot,
    c="b",
    marker="x",
    s=100,
    label="One Hot Encoder DP",
)
axs[0, 0].scatter(
    y=one_hot1["aao"],
    x=one_hot1.auc_tot,
    c="g",
    marker="x",
    s=100,
    label="One Hot Encoder AAO",
)

## No Encoding - Protected attribute is out
axs[0, 0].scatter(
    y=no_encoding1["eof"],
    x=no_encoding1.auc_tot,
    c="r",
    marker="*",
    s=100,
    label="No encoding EOF",
)
axs[0, 0].scatter(
    y=no_encoding1["dp"],
    x=no_encoding1.auc_tot,
    c="b",
    marker="*",
    s=100,
    label="No Encoding DP",
)
axs[0, 0].scatter(
    y=no_encoding1["aao"],
    x=no_encoding1.auc_tot,
    c="g",
    marker="*",
    s=100,
    label="No Encoding AAO",
)

### Figure labels
axs[0, 0].legend()
axs[0, 0].set(xlabel="AUC")
axs[0, 1].set(xlabel="AUC")
axs[0, 0].set(ylabel="Fairness metrics")
axs[0, 1].set_title("Logistic Regression + Smoothing Regularizer")
leg = axs[0, 0].get_legend()
leg.legendHandles[0].set_color("red")
leg.legendHandles[1].set_color("blue")
leg.legendHandles[2].set_color("green")

axs[0, 1].scatter(
    smooth1["auc_tot"].values,
    smooth1["eof"].values,
    s=100,
    c=smooth1.index.values,
    cmap="Reds",
    label="EOF Regularization Parameter (Darker=High)",
)
axs[0, 1].scatter(
    smooth1["auc_tot"].values,
    smooth1["dp"].values,
    s=100,
    c=smooth1.index.values,
    cmap="Blues",
    label="Demographic Parity",
)
axs[0, 1].scatter(
    smooth1["auc_tot"].values,
    smooth1["aao"].values,
    s=100,
    c=smooth1.index.values,
    cmap="Greens",
    label="Average Absolute Odds",
)

### ONE-HOT
axs[0, 1].scatter(
    y=one_hot1["eof"],
    x=one_hot1.auc_tot,
    c="r",
    marker="x",
    s=100,
    label="One Hot Encoder EOF",
)
axs[0, 1].scatter(
    y=one_hot1["dp"],
    x=one_hot1.auc_tot,
    c="b",
    marker="x",
    s=100,
    label="One Hot Encoder DP",
)
axs[0, 1].scatter(
    y=one_hot1["aao"],
    x=one_hot1.auc_tot,
    c="g",
    marker="x",
    s=100,
    label="One Hot Encoder AAO",
)
## No Encoding - Protected attribute is out
axs[0, 1].scatter(
    y=no_encoding1["eof"],
    x=no_encoding1.auc_tot,
    c="r",
    marker="*",
    s=100,
    label="No encoding EOF",
)
axs[0, 1].scatter(
    y=no_encoding1["dp"],
    x=no_encoding1.auc_tot,
    c="b",
    marker="*",
    s=100,
    label="No Encoding DP",
)
axs[0, 1].scatter(
    y=no_encoding1["aao"],
    x=no_encoding1.auc_tot,
    c="g",
    marker="*",
    s=100,
    label="No Encoding AAO",
)
######### DT #########
#######################
axs[1, 0].set_title("Neural Net + Gaussian Noise")
### Fairness metrics plotting
axs[1, 0].scatter(
    gaus2["auc_tot"].values,
    gaus2["eof"].values,
    s=100,
    c=gaus2.index.values,
    cmap="Reds",
    label="EOF Regularization Parameter (Darker=High)",
)
axs[1, 0].scatter(
    gaus2["auc_tot"].values,
    gaus2["dp"].values,
    s=100,
    c=gaus2.index.values,
    cmap="Blues",
    label="Demographic Parity",
)
axs[1, 0].scatter(
    gaus2["auc_tot"].values,
    gaus2["aao"].values,
    s=100,
    c=gaus2.index.values,
    cmap="Greens",
    label="Average Absolute Odds",
)
### ONE-HOT
axs[1, 0].scatter(
    y=one_hot2["eof"],
    x=one_hot2.auc_tot,
    c="r",
    marker="x",
    s=100,
    label="One Hot Encoder EOF",
)
axs[1, 0].scatter(
    y=one_hot2["dp"],
    x=one_hot2.auc_tot,
    c="b",
    marker="x",
    s=100,
    label="One Hot Encoder DP",
)
axs[1, 0].scatter(
    y=one_hot2["aao"],
    x=one_hot2.auc_tot,
    c="g",
    marker="x",
    s=100,
    label="One Hot Encoder AAO",
)

## No Encoding - Protected attribute is out
axs[1, 0].scatter(
    y=no_encoding2["eof"],
    x=no_encoding2.auc_tot,
    c="r",
    marker="*",
    s=100,
    label="No encoding EOF",
)
axs[1, 0].scatter(
    y=no_encoding2["dp"],
    x=no_encoding2.auc_tot,
    c="b",
    marker="*",
    s=100,
    label="No Encoding DP",
)
axs[1, 0].scatter(
    y=no_encoding2["aao"],
    x=no_encoding2.auc_tot,
    c="g",
    marker="*",
    s=100,
    label="No Encoding AAO",
)

### Figure labels
axs[1, 0].legend()
axs[1, 0].set(xlabel="AUC")
axs[1, 1].set(xlabel="AUC")
axs[1, 0].set(ylabel="Fairness metrics")
axs[1, 1].set_title("Neural Net + Smoothing Regularizer")
leg = axs[1, 0].get_legend()
leg.legendHandles[0].set_color("red")
leg.legendHandles[1].set_color("blue")
leg.legendHandles[2].set_color("green")

axs[1, 1].scatter(
    smooth2["auc_tot"].values,
    smooth2["eof"].values,
    s=100,
    c=smooth2.index.values,
    cmap="Reds",
    label="EOF Regularization Parameter (Darker=High)",
)
axs[1, 1].scatter(
    smooth2["auc_tot"].values,
    smooth2["dp"].values,
    s=100,
    c=smooth2.index.values,
    cmap="Blues",
    label="Demographic Parity",
)
axs[1, 1].scatter(
    smooth2["auc_tot"].values,
    smooth2["aao"].values,
    s=100,
    c=smooth2.index.values,
    cmap="Greens",
    label="Average Absolute Odds",
)

### ONE-HOT
axs[1, 1].scatter(
    y=one_hot2["eof"],
    x=one_hot2.auc_tot,
    c="r",
    marker="x",
    s=100,
    label="One Hot Encoder EOF",
)
axs[1, 1].scatter(
    y=one_hot2["dp"],
    x=one_hot2.auc_tot,
    c="b",
    marker="x",
    s=100,
    label="One Hot Encoder DP",
)
axs[1, 1].scatter(
    y=one_hot2["aao"],
    x=one_hot2.auc_tot,
    c="g",
    marker="x",
    s=100,
    label="One Hot Encoder AAO",
)
## No Encoding - Protected attribute is out
axs[1, 1].scatter(
    y=no_encoding2["eof"],
    x=no_encoding2.auc_tot,
    c="r",
    marker="*",
    s=100,
    label="No encoding EOF",
)
axs[1, 1].scatter(
    y=no_encoding2["dp"],
    x=no_encoding2.auc_tot,
    c="b",
    marker="*",
    s=100,
    label="No Encoding DP",
)
axs[1, 1].scatter(
    y=no_encoding2["aao"],
    x=no_encoding2.auc_tot,
    c="g",
    marker="*",
    s=100,
    label="No Encoding AAO",
)

######### GBDT #########
#######################
axs[2, 0].set_title("Gradient Boosting + Gaussian Noise")
### Fairness metrics plotting
axs[2, 0].scatter(
    gaus3["auc_tot"].values,
    gaus3["eof"].values,
    s=100,
    c=gaus3.index.values,
    cmap="Reds",
    label="EOF Regularization Parameter (Darker=High)",
)
axs[2, 0].scatter(
    gaus3["auc_tot"].values,
    gaus3["dp"].values,
    s=100,
    c=gaus3.index.values,
    cmap="Blues",
    label="Demographic Parity",
)
axs[2, 0].scatter(
    gaus3["auc_tot"].values,
    gaus3["aao"].values,
    s=100,
    c=gaus3.index.values,
    cmap="Greens",
    label="Average Absolute Odds",
)
### ONE-HOT
axs[2, 0].scatter(
    y=one_hot3["eof"],
    x=one_hot3.auc_tot,
    c="r",
    marker="x",
    s=100,
    label="One Hot Encoder EOF",
)
axs[2, 0].scatter(
    y=one_hot3["dp"],
    x=one_hot3.auc_tot,
    c="b",
    marker="x",
    s=100,
    label="One Hot Encoder DP",
)
axs[2, 0].scatter(
    y=one_hot3["aao"],
    x=one_hot3.auc_tot,
    c="g",
    marker="x",
    s=100,
    label="One Hot Encoder AAO",
)

## No Encoding - Protected attribute is out
axs[2, 0].scatter(
    y=no_encoding3["eof"],
    x=no_encoding3.auc_tot,
    c="r",
    marker="*",
    s=100,
    label="No encoding EOF",
)
axs[2, 0].scatter(
    y=no_encoding3["dp"],
    x=no_encoding3.auc_tot,
    c="b",
    marker="*",
    s=100,
    label="No Encoding DP",
)
axs[2, 0].scatter(
    y=no_encoding3["aao"],
    x=no_encoding3.auc_tot,
    c="g",
    marker="*",
    s=100,
    label="No Encoding AAO",
)

### Figure labels
axs[2, 0].legend()
axs[2, 0].set(xlabel="AUC")
axs[2, 1].set(xlabel="AUC")
axs[2, 0].set(ylabel="Fairness metrics")
axs[2, 1].set_title("Gradient Boosting + Smoothing Regularizer")
leg = axs[2, 0].get_legend()
leg.legendHandles[0].set_color("red")
leg.legendHandles[1].set_color("blue")
leg.legendHandles[2].set_color("green")

axs[2, 1].scatter(
    smooth3["auc_tot"].values,
    smooth3["eof"].values,
    s=100,
    c=smooth3.index.values,
    cmap="Reds",
    label="EOF Regularization Parameter (Darker=High)",
)
axs[2, 1].scatter(
    smooth3["auc_tot"].values,
    smooth3["dp"].values,
    s=100,
    c=smooth3.index.values,
    cmap="Blues",
    label="Demographic Parity",
)
axs[2, 1].scatter(
    smooth3["auc_tot"].values,
    smooth3["aao"].values,
    s=100,
    c=smooth3.index.values,
    cmap="Greens",
    label="Average Absolute Odds",
)

### ONE-HOT
axs[2, 1].scatter(
    y=one_hot3["eof"],
    x=one_hot3.auc_tot,
    c="r",
    marker="x",
    s=100,
    label="One Hot Encoder EOF",
)
axs[2, 1].scatter(
    y=one_hot3["dp"],
    x=one_hot3.auc_tot,
    c="b",
    marker="x",
    s=100,
    label="One Hot Encoder DP",
)
axs[2, 1].scatter(
    y=one_hot3["aao"],
    x=one_hot3.auc_tot,
    c="g",
    marker="x",
    s=100,
    label="One Hot Encoder AAO",
)
## No Encoding - Protected attribute is out
axs[2, 1].scatter(
    y=no_encoding3["eof"],
    x=no_encoding3.auc_tot,
    c="r",
    marker="*",
    s=100,
    label="No encoding EOF",
)
axs[2, 1].scatter(
    y=no_encoding3["dp"],
    x=no_encoding3.auc_tot,
    c="b",
    marker="*",
    s=100,
    label="No Encoding DP",
)
axs[2, 1].scatter(
    y=no_encoding3["aao"],
    x=no_encoding3.auc_tot,
    c="g",
    marker="*",
    s=100,
    label="No Encoding AAO",
)

fig.savefig("images/encTheoryFull.pdf", bbox_inches="tight")
fig.show()

# %%
## VIS 2 MODELS
########################
########################

fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=True)
######### LR #########
########################
axs[0, 0].set_title("Logistic Regression + Gaussian Noise")
### Fairness metrics plotting
axs[0, 0].scatter(
    gaus1["auc_tot"].values,
    gaus1["eof"].values,
    s=100,
    c=gaus1.index.values,
    cmap="Reds",
    label="EOF Regularization Parameter (Darker=High)",
)
axs[0, 0].scatter(
    gaus1["auc_tot"].values,
    gaus1["dp"].values,
    s=100,
    c=gaus1.index.values,
    cmap="Blues",
    label="Demographic Parity",
)
axs[0, 0].scatter(
    gaus1["auc_tot"].values,
    gaus1["aao"].values,
    s=100,
    c=gaus1.index.values,
    cmap="Greens",
    label="Average Absolute Odds",
)
### ONE-HOT
axs[0, 0].scatter(
    y=one_hot1["eof"],
    x=one_hot1.auc_tot,
    c="r",
    marker="x",
    s=100,
    label="One Hot Encoder EOF",
)
axs[0, 0].scatter(
    y=one_hot1["dp"],
    x=one_hot1.auc_tot,
    c="b",
    marker="x",
    s=100,
    label="One Hot Encoder DP",
)
axs[0, 0].scatter(
    y=one_hot1["aao"],
    x=one_hot1.auc_tot,
    c="g",
    marker="x",
    s=100,
    label="One Hot Encoder AAO",
)

## No Encoding - Protected attribute is out
axs[0, 0].scatter(
    y=no_encoding1["eof"],
    x=no_encoding1.auc_tot,
    c="r",
    marker="*",
    s=100,
    label="No encoding EOF",
)
axs[0, 0].scatter(
    y=no_encoding1["dp"],
    x=no_encoding1.auc_tot,
    c="b",
    marker="*",
    s=100,
    label="No Encoding DP",
)
axs[0, 0].scatter(
    y=no_encoding1["aao"],
    x=no_encoding1.auc_tot,
    c="g",
    marker="*",
    s=100,
    label="No Encoding AAO",
)

### Figure labels
axs[0, 0].legend()
axs[0, 0].set(xlabel="AUC")
axs[0, 1].set(xlabel="AUC")
axs[0, 0].set(ylabel="Fairness metrics")
axs[0, 1].set_title("Logistic Regression + Smoothing Regularizer")
leg = axs[0, 0].get_legend()
leg.legendHandles[0].set_color("red")
leg.legendHandles[1].set_color("blue")
leg.legendHandles[2].set_color("green")

axs[0, 1].scatter(
    smooth1["auc_tot"].values,
    smooth1["eof"].values,
    s=100,
    c=smooth1.index.values,
    cmap="Reds",
    label="EOF Regularization Parameter (Darker=High)",
)
axs[0, 1].scatter(
    smooth1["auc_tot"].values,
    smooth1["dp"].values,
    s=100,
    c=smooth1.index.values,
    cmap="Blues",
    label="Demographic Parity",
)
axs[0, 1].scatter(
    smooth1["auc_tot"].values,
    smooth1["aao"].values,
    s=100,
    c=smooth1.index.values,
    cmap="Greens",
    label="Average Absolute Odds",
)

### ONE-HOT
axs[0, 1].scatter(
    y=one_hot1["eof"],
    x=one_hot1.auc_tot,
    c="r",
    marker="x",
    s=100,
    label="One Hot Encoder EOF",
)
axs[0, 1].scatter(
    y=one_hot1["dp"],
    x=one_hot1.auc_tot,
    c="b",
    marker="x",
    s=100,
    label="One Hot Encoder DP",
)
axs[0, 1].scatter(
    y=one_hot1["aao"],
    x=one_hot1.auc_tot,
    c="g",
    marker="x",
    s=100,
    label="One Hot Encoder AAO",
)
## No Encoding - Protected attribute is out
axs[0, 1].scatter(
    y=no_encoding1["eof"],
    x=no_encoding1.auc_tot,
    c="r",
    marker="*",
    s=100,
    label="No encoding EOF",
)
axs[0, 1].scatter(
    y=no_encoding1["dp"],
    x=no_encoding1.auc_tot,
    c="b",
    marker="*",
    s=100,
    label="No Encoding DP",
)
axs[0, 1].scatter(
    y=no_encoding1["aao"],
    x=no_encoding1.auc_tot,
    c="g",
    marker="*",
    s=100,
    label="No Encoding AAO",
)
######### GBDT #########
#######################
axs[1, 0].set_title("Gradient Boosting + Gaussian Noise")
### Fairness metrics plotting
axs[1, 0].scatter(
    gaus3["auc_tot"].values,
    gaus3["eof"].values,
    s=100,
    c=gaus3.index.values,
    cmap="Reds",
    label="EOF Regularization Parameter (Darker=High)",
)
axs[1, 0].scatter(
    gaus3["auc_tot"].values,
    gaus3["dp"].values,
    s=100,
    c=gaus3.index.values,
    cmap="Blues",
    label="Demographic Parity",
)
axs[1, 0].scatter(
    gaus3["auc_tot"].values,
    gaus3["aao"].values,
    s=100,
    c=gaus3.index.values,
    cmap="Greens",
    label="Average Absolute Odds",
)
### ONE-HOT
axs[1, 0].scatter(
    y=one_hot3["eof"],
    x=one_hot3.auc_tot,
    c="r",
    marker="x",
    s=100,
    label="One Hot Encoder EOF",
)
axs[1, 0].scatter(
    y=one_hot3["dp"],
    x=one_hot3.auc_tot,
    c="b",
    marker="x",
    s=100,
    label="One Hot Encoder DP",
)
axs[1, 0].scatter(
    y=one_hot3["aao"],
    x=one_hot3.auc_tot,
    c="g",
    marker="x",
    s=100,
    label="One Hot Encoder AAO",
)

## No Encoding - Protected attribute is out
axs[1, 0].scatter(
    y=no_encoding3["eof"],
    x=no_encoding3.auc_tot,
    c="r",
    marker="*",
    s=100,
    label="No encoding EOF",
)
axs[1, 0].scatter(
    y=no_encoding3["dp"],
    x=no_encoding3.auc_tot,
    c="b",
    marker="*",
    s=100,
    label="No Encoding DP",
)
axs[1, 0].scatter(
    y=no_encoding3["aao"],
    x=no_encoding3.auc_tot,
    c="g",
    marker="*",
    s=100,
    label="No Encoding AAO",
)

### Figure labels
axs[1, 0].legend()
axs[1, 0].set(xlabel="AUC")
axs[1, 1].set(xlabel="AUC")
axs[1, 0].set(ylabel="Fairness metrics")
axs[1, 1].set_title("Gradient Boosting + Smoothing Regularizer")
leg = axs[1, 0].get_legend()
leg.legendHandles[0].set_color("red")
leg.legendHandles[1].set_color("blue")
leg.legendHandles[2].set_color("green")

axs[1, 1].scatter(
    smooth3["auc_tot"].values,
    smooth3["eof"].values,
    s=100,
    c=smooth3.index.values,
    cmap="Reds",
    label="EOF Regularization Parameter (Darker=High)",
)
axs[1, 1].scatter(
    smooth3["auc_tot"].values,
    smooth3["dp"].values,
    s=100,
    c=smooth3.index.values,
    cmap="Blues",
    label="Demographic Parity",
)
axs[1, 1].scatter(
    smooth3["auc_tot"].values,
    smooth3["aao"].values,
    s=100,
    c=smooth3.index.values,
    cmap="Greens",
    label="Average Absolute Odds",
)

### ONE-HOT
axs[1, 1].scatter(
    y=one_hot3["eof"],
    x=one_hot3.auc_tot,
    c="r",
    marker="x",
    s=100,
    label="One Hot Encoder EOF",
)
axs[1, 1].scatter(
    y=one_hot3["dp"],
    x=one_hot3.auc_tot,
    c="b",
    marker="x",
    s=100,
    label="One Hot Encoder DP",
)
axs[1, 1].scatter(
    y=one_hot3["aao"],
    x=one_hot3.auc_tot,
    c="g",
    marker="x",
    s=100,
    label="One Hot Encoder AAO",
)
## No Encoding - Protected attribute is out
axs[1, 1].scatter(
    y=no_encoding3["eof"],
    x=no_encoding3.auc_tot,
    c="r",
    marker="*",
    s=100,
    label="No encoding EOF",
)
axs[1, 1].scatter(
    y=no_encoding3["dp"],
    x=no_encoding3.auc_tot,
    c="b",
    marker="*",
    s=100,
    label="No Encoding DP",
)
axs[1, 1].scatter(
    y=no_encoding3["aao"],
    x=no_encoding3.auc_tot,
    c="g",
    marker="*",
    s=100,
    label="No Encoding AAO",
)

fig.savefig("images/enc2modelsCompass.pdf", bbox_inches="tight")

fig.show()

# %%
