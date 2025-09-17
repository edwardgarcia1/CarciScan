from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import re
import pandas as pd  # Added missing import
import os
from scipy.sparse import hstack

VERBOSE = False

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

if __name__ == "__main__":
    # 1. Load dataset
    dataset = os.path.join(os.getcwd(), '../../dataset/all_toxin_data_fixed.csv')
    print(f"Attempting to load dataset from: {dataset}")
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"Dataset not found at {dataset}.")
    df = pd.read_csv(dataset)

    # 2. Clean and normalize labels
    df['Normalized_Label'] = df['carcinogenicity'].apply(normalize_label)

    df = df[df['Normalized_Label'] != "Not listed by IARC"].reset_index(drop=True)

    # Encode labels
    label_encoder = LabelEncoder()
    df["carcinogenic_numeric"] = label_encoder.fit_transform(df["Normalized_Label"])
    labels = df["carcinogenic_numeric"]

    print("Final Class Distribution:")
    for i, class_name in enumerate(label_encoder.classes_):
        count = (df["carcinogenic_numeric"] == i).sum()
        print(f"{class_name}: {count}")
    print()

    # 3. Select relevant features for model input
    relevant_text_cols = [
        'common_name', 'title', 'description', 'synonyms_list',
        'route_of_exposure', 'mechanism_of_toxicity', 'metabolism', 'toxicity',
        'use_source', 'min_risk_level', 'health_effects', 'symptoms', 'treatment',
        'types', 'cellular_locations', 'tissues', 'pathways'
    ]
    relevant_numeric_cols = [
        'weight', 'melting_point', 'boiling_point', 'solubility', 'logp', 'lethaldose'
    ]
    relevant_formula_cols = [
        'chemical_formula', 'moldb_smiles', 'moldb_formula', 'moldb_inchi', 'moldb_inchikey'
    ]
    # Combine all relevant text columns into a single string
    df['combined_text'] = df[relevant_text_cols + relevant_formula_cols].fillna('').astype(str).agg(' '.join, axis=1)
    # Ensure all relevant numeric columns exist in the DataFrame
    for col in relevant_numeric_cols:
        if col not in df.columns:
            df[col] = 0.0  # or np.nan, but 0.0 is safe for scaling
    # Convert all values to numeric, coerce errors to NaN
    for col in relevant_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Fill all NaNs with the median of each column, or 0 if median is NaN
    for col in relevant_numeric_cols:
        median = df[col].median()
        if np.isnan(median):
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = df[col].fillna(median)
    # As a last resort, fill any remaining NaNs with 0
    if df[relevant_numeric_cols].isnull().any().any():
        print('Warning: Forcibly filling remaining NaNs in numeric features with 0!')
        df[relevant_numeric_cols] = df[relevant_numeric_cols].fillna(0.0)
    # Scale numeric features
    if relevant_numeric_cols:
        scaler = StandardScaler()
        df[relevant_numeric_cols] = scaler.fit_transform(df[relevant_numeric_cols])
    # 4. TF-IDF for text features
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_text = vectorizer.fit_transform(df['combined_text'])
    # 5. Combine text and numeric features
    if relevant_numeric_cols:
        X_numeric = df[relevant_numeric_cols].values
        from scipy import sparse
        X = hstack([X_text, sparse.csr_matrix(X_numeric)])
    else:
        X = X_text

    # 6. Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 7. Handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # 8. Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,  # let it grow deep
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    # 9. Predict
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)

    # 10. Evaluation
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec, rec, f1, supp = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("Multi-Class Metrics (Random Forest):")
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

def predict_ingredient_carcinogenicity(ingredient_info, vectorizer, rf_model, label_encoder, relevant_text_cols, relevant_numeric_cols, relevant_formula_cols):
    """
    Predict carcinogenicity for a new ingredient.
    ingredient_info: dict with keys matching relevant columns
    Returns: predicted label (string)
    """
    # Prepare text
    text_fields = [ingredient_info.get(col, '') for col in relevant_text_cols + relevant_formula_cols]
    combined_text = ' '.join([str(x) for x in text_fields])
    X_text = vectorizer.transform([combined_text])
    # Prepare numeric
    X_numeric = []
    for col in relevant_numeric_cols:
        val = ingredient_info.get(col, None)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
        X_numeric.append(val)
    X_numeric = np.array(X_numeric).reshape(1, -1)
    X = X_text
    if relevant_numeric_cols:
        X = sparse.hstack([X_text, sparse.csr_matrix(X_numeric)])
    pred_numeric = rf_model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_numeric])[0]
    return pred_label

# Example usage (uncomment to test):
# ingredient = {
#     'common_name': 'Caffeine',
#     'title': 'Caffeine',
#     'description': 'A stimulant found in coffee and tea.',
#     'synonyms_list': '1,3,7-Trimethylxanthine',
#     'weight': 194.19,
#     'melting_point': 238,
#     'boiling_point': '',
#     'solubility': '',
#     'logp': -0.07,
#     'lethaldose': '',
#     # ... other fields as needed ...
# }
# print(predict_ingredient_carcinogenicity(ingredient, vectorizer, rf_model, label_encoder, relevant_text_cols, relevant_numeric_cols, relevant_formula_cols))
