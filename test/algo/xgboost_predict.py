import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
import xgboost as xgb
import numpy as np
import re
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

VERBOSE = False
TRAIN_VERBOSE = False  # Set to True to see per-iteration (boosting round) logs


def normalize_label(label):
    """
    Normalize raw dataset labels into standardized IARC groups:
    - Group 1
    - Group 2A
    - Group 2B
    - Group 3
    - No indication
    - Not listed by IARC
    """
    text = str(label).lower().strip()

    # Remove references in parentheses like (L135), (A2976)
    text = re.sub(r"\(.*?\)", "", text).strip()

    if "group 2a" in text or "probably carcinogenic" in text or "2a" in text:
        return "Group 2A"
    elif "group 2b" in text or "possibly carcinogenic" in text or "2b" in text:
        return "Group 2B"
    elif "group 3" in text or "not classifiable" in text or "3" in text:
        return "Group 3"
    elif "group 1" in text or "carcinogenic to humans" in text or "1," in text:
        return "Group 1"
    elif "no indication" in text:
        return "No indication"
    # elif "oncogenesis" in text or "oncogenic" in text:
    #     return "Suspected oncogenic"
    # elif "metabolite" in text:
    #     return "Metabolite (related to listed substance)"
    # elif "related pahs" in text or "other pahs" in text:
    #     return "Related PAHs (not listed)"
    # elif "genetic damage" in text:
    #     return "May cause genetic damage (not listed)"
    # elif "class b carcinogen" in text:
    #     return "Class B carcinogen (EPA)"
    else:
        return "Not listed by IARC"


def preprocess_common_name(name):
    """
    Normalize the common name: lowercase, remove punctuation, strip whitespace.
    """
    name = str(name).lower()
    name = re.sub(r"[^\w\s]", " ", name)  # Remove punctuation
    name = re.sub(r"\s+", " ", name)  # Collapse multiple spaces
    return name.strip()


def expand_with_synonyms(df):
    """
    Expand the dataframe so each variant and synonym is a separate row with the same label.
    Optionally concatenate description for richer context.
    Filter out ambiguous/short synonyms.
    """
    expanded_rows = []
    for _, row in df.iterrows():
        label = row['Normalized_Label']
        description = str(row.get('description', '')).strip()
        # Add common_name
        name = preprocess_common_name(row['common_name'])
        if name and len(name) > 2:
            text = name
            expanded_rows.append({'name': text, 'label': label})
        # Add synonyms
        synonyms = str(row.get('synonyms_list', '')).strip()
        if synonyms and synonyms.lower() != 'nan':
            for syn in re.split(r'[;,]', synonyms):
                syn = preprocess_common_name(syn)
                # Filter out short/ambiguous synonyms
                if syn and syn != name and len(syn) > 2 and syn not in ['ion', 'compound', 'cation', 'anion']:
                    text = syn + ' ' + description if description else syn
                    expanded_rows.append({'name': text, 'label': label})
    return pd.DataFrame(expanded_rows)


# 1. Load dataset (use same path logic as random_forest.py)
dataset = os.path.join(os.getcwd(), '../../dataset/all_toxin_data_fixed.csv')
print(f"Attempting to load dataset from: {dataset}")
if not os.path.exists(dataset):
    raise FileNotFoundError(f"Dataset not found at {dataset}.")
df = pd.read_csv(dataset)

# 2. Clean and normalize labels
# Apply normalization to clean the labels
df['Normalized_Label'] = df['carcinogenicity'].apply(normalize_label)
# Filter out rows labeled as "Not listed by IARC"
df = df[df['Normalized_Label'] != "Not listed by IARC"].reset_index(drop=True)

# Expand dataset with synonyms and description
expanded_df = expand_with_synonyms(df)

# Encode labels
label_encoder = LabelEncoder()
expanded_df["carcinogenic_numeric"] = label_encoder.fit_transform(expanded_df["label"])
labels = expanded_df["carcinogenic_numeric"]

print("Final Class Distribution:")
for i, class_name in enumerate(label_encoder.classes_):
    count = (expanded_df["carcinogenic_numeric"] == i).sum()
    print(f"{class_name}: {count}")
print()

# Preprocess and vectorize names only (with char n-grams)
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, analyzer='char_wb', ngram_range=(2,5))
X = vectorizer.fit_transform(expanded_df['name'])

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

# Handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
sample_weights = np.array([class_weight_dict[y] for y in y_train])

# Train XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=None,
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    random_state=42,
    n_jobs=-1,
    verbosity=1 if TRAIN_VERBOSE else 0
)
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

# Predict
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
prec, rec, f1, supp = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("Multi-Class Metrics (XGBoost):")
print(f"  Accuracy: {acc:.4f}")
print(f"  Balanced Accuracy: {bal_acc:.4f}")

# ROC-AUC (multi-class)
try:
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    print(f"  ROC-AUC (One-vs-Rest): {roc_auc:.4f}")
except Exception as e:
    print(f"  ROC-AUC: Could not calculate - {str(e)}")

# Per-class metrics
class_names = label_encoder.classes_
for i, class_name in enumerate(class_names):
    print(f"  {class_name}:")
    print(f"    Precision: {prec[i]:.4f}")
    print(f"    Recall:    {rec[i]:.4f}")
    print(f"    F1-score:  {f1[i]:.4f}")
    print(f"    Support:   {int(supp[i])}")

print("  Confusion Matrix:")
print(f"    {cm.tolist()}")
print("  Class mapping:")
for i, class_name in enumerate(class_names):
    print(f"    {i}: {class_name}")

# Print confusion matrix and sample misclassifications for error analysis
print("Confusion Matrix:")
print(cm)
print("Sample misclassifications:")
for i in range(min(10, len(y_test))):
    if y_pred[i] != y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]:
        print(f"Name: {expanded_df.iloc[X_test.indices[i]]['name'] if hasattr(X_test, 'indices') else 'N/A'} | True: {label_encoder.classes_[y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]]} | Pred: {label_encoder.classes_[y_pred[i]]}")

# Optional full report (verbose only)
if VERBOSE:
    print("\nFull classification report (sklearn):")
    print(classification_report(y_test, y_pred, zero_division=0))

# 9. Show probability summary (verbose)
if VERBOSE:
    print(f"\nPredicted probabilities for test set:")
    print(f"Min probability: {y_pred_proba.min():.4f}")
    print(f"Max probability: {y_pred_proba.max():.4f}")
    print(f"Mean probability: {y_pred_proba.mean():.4f}")

    carcinogenic_indices = np.where(y_test == 1)[0]
    if len(carcinogenic_indices) > 0:
        print(f"\nPredictions for actual carcinogenic compounds:")
        for idx in carcinogenic_indices:
            true_label = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
            pred_label = y_pred[idx]
            prob = y_pred_proba[idx]
            print(f"  True: {true_label}, Predicted: {pred_label}, Probability: {prob:.4f}")
