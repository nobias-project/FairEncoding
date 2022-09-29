# %%
import pandas as pd
import random

random.seed(0)
from collections import defaultdict

pd.set_option("display.max_columns", None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance

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
    auc_group,
    fit_predict,
    metric_calculator,
    plot_rolling,
    scale_output,
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
data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
try:
    ca_data = data_source.get_data(states=["CA"], download=False)
except:
    ca_data = data_source.get_data(states=["CA"], download=True)
ca_features, ca_labels, ca_group = ACSIncome.df_to_numpy(ca_data)

# Preprocesssing
## Scale & Conver to DF
ca_features = StandardScaler().fit_transform(ca_features)
ca_features = pd.DataFrame(ca_features, columns=ACSIncome.features)
ca_features = ca_features.drop(columns=["RAC1P"])
ca_features["group"] = ca_group
ca_features["label"] = ca_labels
## Encode race back as str
race = {
    1: "White",
    2: "Black",
    3: "Native",
    4: "Alaska",
    5: "AmericanIndian",
    6: "Asian",
    7: "Hawaiian",
    8: "Other",
}
ca_features["group"] = ca_features["group"].map(race)
# Analysis
print(ca_features.groupby("group").label.mean())
print(ca_features.groupby("group").label.count())
# %%
# Remove groups that have small statistical mass
# ca_features = ca_features[(ca_features["group"] == 1) | (ca_features["group"] == 2)]

# Auxiliary data for plottign
aux = pd.DataFrame(ca_features["group"].value_counts())


def func(pct, allvals):
    absolute = int(np.round(pct / 100.0 * np.sum(allvals)))
    return "{:.1f}%\n({:d})".format(pct, absolute)


colors = sns.color_palette("pastel")[0 : aux.shape[0]]
# create pie chart
plt.figure()
explode = (0.05,) * aux.shape[0]
plt.pie(
    aux.group.values,
    labels=aux.index,
    autopct=lambda pct: func(pct, aux.group.values),
    shadow=True,
    explode=explode,
)
plt.show()

# %%
X = ca_features.drop(columns=["label"])
y = ca_features[["label"]]
# In[11]:
for col in X.columns:
    print(X[col].value_counts())
for col in X.columns:
    print(len(X[col].unique()))


# %%
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=42)

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


def explain(xgb: bool = True):
    """
    Provide a SHAP explanation by fitting MEstimate and GBDT
    """
    if xgb:
        pipe = Pipeline(
            [("encoder", MEstimateEncoder()), ("model", GradientBoostingClassifier())]
        )
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
    modelo, data: pd.DataFrame, truth: pd.DataFrame, col: str, group1: str, group2: str
):
    """
    model: model to be used
    data: data to predict
    truth: ground truth labels
    col: column to be used as group
    group1: Reference group
    group2: Discriminated group
    """
    aux = data.copy()
    aux["target"] = truth

    # Filter the data
    g1 = data[data[col] == group1]
    g2 = data[data[col] == group2]

    # Filter the ground truth
    g1_true = aux[aux[col] == group1].target
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

    return eof, dp, aao


# %%
# Train model
m = Pipeline([("enc", CatBoostEncoder(sigma=0.5)), ("model", LogisticRegression())])
m.fit(X_tr, y_tr)
roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])

# %%
res = {}
for cat, num in X["group"].value_counts().items():
    COL = "group"
    GROUP1 = "White"
    GROUP2 = cat
    res[cat] = [
        metric_calculator(
            modelo=m, data=X, truth=y, col=COL, group1=GROUP1, group2=GROUP2
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
    res["eof"] = res["fairness_metric"].apply(lambda x: x[0])
    res["dp"] = res["fairness_metric"].apply(lambda x: x[1])
    res["aao"] = res["fairness_metric"].apply(lambda x: x[2])
    res = res.drop(columns="fairness_metric")
    res["auc_tot"] = auc_tot
    res["auc_" + GROUP1] = auc[GROUP1]
    res["auc_" + GROUP2] = auc[GROUP2]

    return res


# %%
# Experiment parameters
COL = "group"
GROUP1 = "White"
GROUP2 = "Other"

# %%
## LR Experiment
one_hot1 = fair_encoder(model=LogisticRegression(), enc="ohe", param=[0])

PARAM = np.linspace(0, 1, 20)
gaus1 = fair_encoder(
    model=LogisticRegression(),
    enc="catboost",
    param=PARAM,
)
PARAM = np.linspace(0, 100_000, 20)
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
axs[0].scatter(y=one_hot1["eof"], x=one_hot1.auc_tot, s=100, label="One Hot Encoder")

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

axs[1].scatter(y=one_hot1["eof"], x=one_hot1.auc_tot, s=100, label="One Hot Encoder")
fig.savefig("images/encTheory.png")
fig.show()
# %%
# %%
##################
### Figure 2 #####
##################

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

aux = gaus1[["dp"]].rolling(5).mean().dropna()

axs[1].plot(aux["fairness_metric"], label=GROUP1 + " vs " + GROUP2, color="r")

axs[1].legend()
axs[1].set_title("Fairness Metric")
axs[1].set_ylabel("Equal opportunity fairness (TPR)")
axs[1].set_xlabel("Regularization parameter")
plt.savefig("images/compassHyperGaussian.png")
plt.show()


# In[35]:
fig, axs = plt.subplots(1, 2, sharex=True)

fig.suptitle("Smoothing regularization target encoding")
aux = smooth1[["auc_tot", "auc_White", "auc_Asian"]].rolling(5).mean().dropna()


for col in aux.columns:
    axs[0].plot(aux[col], label=col)
    # plt.fill_between(aux.index,(aux[col] - stand[col]),(aux[col] + stand[col]),# color="b",alpha=0.1,)
axs[0].legend()
axs[0].set_title("Model performance")
axs[0].set_ylabel("AUC")
axs[0].set_xlabel("Regularization parameter")

aux = smooth1[["dp"]].rolling(5).mean().dropna()

axs[1].plot(aux["dp"], label=GROUP1 + " vs " + GROUP2, color="r")

axs[1].legend()
axs[1].set_title("Fairness Metric")
axs[1].set_ylabel("Equal opportunity fairness (TPR)")
axs[1].set_xlabel("Regularization parameter")
plt.savefig("images/compassHyperSmoothing.png")
plt.show()


# In[31]:


## LR
# one_hot1 = fair_encoder(model=LogisticRegression(), enc="ohe", param=[0])

# PARAM = np.linspace(0, 1, 50)
# gaus1 = fair_encoder(model=LogisticRegression(), enc="catboost", param=PARAM,un_regularize=cols_enc)
# PARAM = np.linspace(0, 100, 50)
# smooth1 = fair_encoder(model=LogisticRegression(), enc="mestimate", param=PARAM,un_regularize=cols_enc)
## DT
one_hot2 = fair_encoder(model=DecisionTreeClassifier(max_depth=5), enc="ohe", param=[0])
PARAM = np.linspace(0, 1, 50)
gaus2 = fair_encoder(
    model=DecisionTreeClassifier(max_depth=5),
    enc="catboost",
    param=PARAM,
    un_regularize=cols_enc,
)
PARAM = np.linspace(0, 100_000, 50)
smooth2 = fair_encoder(
    model=DecisionTreeClassifier(max_depth=5),
    enc="mestimate",
    param=PARAM,
    un_regularize=cols_enc,
)
## GBDT
one_hot3 = fair_encoder(model=GradientBoostingClassifier(), enc="ohe", param=[0])

PARAM = np.linspace(0, 1, 50)
gaus3 = fair_encoder(
    model=GradientBoostingClassifier(),
    enc="catboost",
    param=PARAM,
    un_regularize=cols_enc,
)
PARAM = np.linspace(0, 100_000, 50)
smooth3 = fair_encoder(
    model=GradientBoostingClassifier(),
    enc="mestimate",
    param=PARAM,
    un_regularize=cols_enc,
)


# In[36]:


fig, axs = plt.subplots(3, 2, figsize=(15, 15), sharex=True, sharey=True)

# LR
axs[0, 0].set_title("Logistic Regression + Gaussian Noise")
# axs[0, 0].axis(xmin=0.5,xmax=13.5)
axs[0, 0].scatter(
    gaus1["auc_tot"].values,
    gaus1["fairness_metric"].values,
    s=100,
    c=gaus1.index.values,
    cmap="Reds",
    label="Regularization Parameter (Darker=High)",
)
axs[0, 0].scatter(
    y=one_hot1.fairness_metric, x=one_hot1.auc_tot, s=100, label="One Hot Encoder"
)
axs[0, 0].legend()
axs[0, 1].set_title("Logistic Regression + Smoothing Regularizer")
axs[0, 1].scatter(
    smooth1["auc_tot"].values,
    smooth1["fairness_metric"].values,
    s=100,
    c=smooth1.index.values,
    cmap="Reds",
    label="Regularization Parameter (Darker=High)",
)
axs[0, 1].scatter(
    y=one_hot1.fairness_metric, x=one_hot1.auc_tot, s=100, label="One Hot Encoder"
)

## DT
axs[1, 0].set_title("Decision Tree + Gaussian Noise")
axs[1, 0].set(ylabel="Equal opportunity fairness (TPR)")
axs[1, 0].scatter(
    gaus2["auc_tot"].values,
    gaus2["fairness_metric"].values,
    s=100,
    c=gaus2.index.values,
    cmap="Reds",
    label="Regularization Parameter (Darker=High)",
)
axs[1, 0].scatter(
    y=one_hot2.fairness_metric, x=one_hot2.auc_tot, s=100, label="One Hot Encoder"
)

axs[1, 1].set_title("Decision Tree + Smoothing Regularizer")
axs[1, 1].scatter(
    smooth2["auc_tot"].values,
    smooth2["fairness_metric"].values,
    s=100,
    c=smooth2.index.values,
    cmap="Reds",
    label="Regularization Parameter (Darker=High)",
)
axs[1, 1].scatter(
    y=one_hot2.fairness_metric, x=one_hot2.auc_tot, s=100, label="One Hot Encoder"
)

# GBDT
axs[2, 0].set_title("Gradient Boosting + Gaussian Noise")
axs[2, 0].set(xlabel="AUC")
axs[2, 0].scatter(
    gaus3["auc_tot"].values,
    gaus3["fairness_metric"].values,
    s=100,
    c=gaus3.index.values,
    cmap="Reds",
    label="Regularization Parameter (Darker=High)",
)
axs[2, 0].scatter(
    y=one_hot3.fairness_metric, x=one_hot3.auc_tot, s=100, label="One Hot Encoder"
)

axs[2, 1].set_title("Gradient Boosting + Smoothing Regularizer")
axs[2, 1].set(xlabel="AUC")
axs[2, 1].scatter(
    smooth3["auc_tot"].values,
    smooth3["fairness_metric"].values,
    s=100,
    c=smooth3.index.values,
    cmap="Reds",
    label="Regularization Parameter (Darker=High)",
)
axs[2, 1].scatter(
    y=one_hot3.fairness_metric, x=one_hot3.auc_tot, s=100, label="One Hot Encoder"
)

fig.savefig("images/encTheoryFull.png")
fig.show()


# In[ ]:
