!pip install shap scikit-learn pandas numpy matplotlib seaborn ipywidgets xgboost fpdf --quiet

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from scipy.stats import pearsonr, f_oneway
from fpdf import FPDF
import io, warnings


# ==============================================================================
# 1. DATA GENERATION (TRAINING DATA – REGULATOR-GRADE PROXIES)
# ==============================================================================

N = 5000
df = pd.DataFrame({
    "GST_Consistency": np.random.beta(7,2,N),
    "Debt_Equity": np.random.normal(1.8,0.6,N),
    "ROA": np.random.normal(0.08,0.03,N),
    "Physical_Risk_Index": np.random.beta(3,4,N)*100,
    "Green_Capex_Score": np.random.choice([0,1],N,p=[0.26,0.74])*np.random.normal(25,10,N),
    "NLP_Authenticity_Score": np.random.normal(0.65,0.15,N).clip(0,1),
    "Digital_Readiness": np.random.normal(4.2,0.6,N).clip(1,5),
    "Technostress": np.random.normal(2.9,0.7,N).clip(1,5),
    "Region": np.random.choice(["South","North","East","West"],N,p=[0.38,0.22,0.20,0.20])
})

risk = (
    (1-df["GST_Consistency"])*40 +
    (df["Physical_Risk_Index"]/100)*30 +
    (1-df["NLP_Authenticity_Score"])*10 +
    (df["Debt_Equity"]/3)*10 -
    df["Digital_Readiness"]*5
)

df["Default"] = (1/(1+np.exp(-(risk-risk.mean())))>0.8).astype(int)

df_model = pd.get_dummies(df, drop_first=True)

# ==============================================================================
# 2. STATISTICAL VALIDATION (YOUR PAPER NUMERICALS)
# ==============================================================================

print("\nMEANS")
print(df[["Digital_Readiness","Technostress"]].mean())

print("\nCORRELATIONS")
print("Digital ↔ Green Capex:", pearsonr(df["Digital_Readiness"],df["Green_Capex_Score"]))
print("Technostress ↔ Green Capex:", pearsonr(df["Technostress"],df["Green_Capex_Score"]))

print("\nANOVA (Regional Digital Readiness)")
anova = f_oneway(*[df[df["Region"]==r]["Digital_Readiness"] for r in df["Region"].unique()])
print(anova)

# ==============================================================================
# 3. HYBRID MODEL (RF + XGBOOST)
# ==============================================================================

X = df_model.drop("Default",axis=1)
y = df_model["Default"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

rf = RandomForestClassifier(n_estimators=300,max_depth=12,random_state=42)
xgb = XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=6,
                    eval_metric="logloss",use_label_encoder=False)

hybrid = VotingClassifier([("rf",rf),("xgb",xgb)],voting="soft")
hybrid.fit(X_train,y_train)
rf.fit(X_train,y_train)

print("\nMODEL ACCURACY")
print("Hybrid RF + XGB:", accuracy_score(y_test,hybrid.predict(X_test)))

# ==============================================================================
# 4. SHAP (STABLE)
# ==============================================================================

explainer = shap.TreeExplainer(rf)
sv = explainer(X_test)
shap.summary_plot(sv.values[:,:,1],X_test)

# ==============================================================================
# 5. BATCH MSME AUDIT — FILEUPLOAD (NO PATH ISSUES)
# ==============================================================================

print("\n" + "="*70)
print("PORTFOLIO MODE: UPLOAD MSME CSV (100 UNITS)")
print("="*70)

from google.colab import files

uploaded = files.upload()
filename = list(uploaded.keys())[0]
df_batch = pd.read_csv(io.BytesIO(uploaded[filename]))

out = widgets.Output()

def run_batch(change):
    with out:
        clear_output()
        if not uploader.value:
            print("Upload a CSV file.")
            return

        file = next(iter(uploader.value.values()))
        df_batch = pd.read_csv(io.BytesIO(file["content"]))

        required = [
            "GST_Consistency","Physical_Risk_Index","Green_Capex_Score",
            "NLP_Authenticity_Score","Debt_Equity","ROA",
            "Digital_Readiness","Technostress"
        ]
        missing = [c for c in required if c not in df_batch.columns]
        if missing:
            print("Missing columns:", missing)
            return

        Xb = pd.get_dummies(df_batch)
        Xb = Xb.reindex(columns=X.columns,fill_value=0)

        probs = hybrid.predict_proba(Xb)[:,1]
        df_batch["Default_Probability"] = probs
        df_batch["Decision"] = np.where(probs<0.35,"APPROVED","REJECTED")

        print(f"TOTAL MSMEs: {len(df_batch)}")
        print(f"APPROVED: {(df_batch['Decision']=='APPROVED').sum()}")
        print(f"REJECTED: {(df_batch['Decision']=='REJECTED').sum()}")

        # HISTOGRAM
        plt.figure(figsize=(10,5))
        sns.histplot(df_batch,x="Default_Probability",hue="Decision",
                     bins=25,multiple="stack",
                     palette={"APPROVED":"green","REJECTED":"red"})
        plt.axvline(0.35,color="black",linestyle="--")
        plt.title("Portfolio Default Risk Distribution")
        plt.show()

updf_batch = pd.read_csv("draft2.csv")

Xb = pd.get_dummies(df_batch)
Xb = Xb.reindex(columns=X.columns, fill_value=0)

probs = hybrid.predict_proba(Xb)[:,1]
df_batch["Default_Probability"] = probs
df_batch["Decision"] = np.where(probs < 0.35, "APPROVED", "REJECTED")
import matplotlib.pyplot as plt
import numpy as np

approved = df_batch[df_batch["Decision"] == "APPROVED"]["Default_Probability"]
rejected = df_batch[df_batch["Decision"] == "REJECTED"]["Default_Probability"]

plt.figure(figsize=(12,6))

plt.hist(
    approved,
    bins=25,
    alpha=0.9,
    label="APPROVED",
    edgecolor="black"
)

plt.hist(
    rejected,
    bins=25,
    alpha=0.9,
    label="REJECTED",
    edgecolor="black"
)

plt.axvline(
    x=0.35,
    linestyle="--",
    linewidth=2,
    label="Approval Threshold (0.35)"
)

plt.title("Portfolio Risk Distribution", fontsize=14)
plt.xlabel("Predicted Default Probability")
plt.ylabel("Count of MSMEs")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()


print(df_batch["Decision"].value_counts())
