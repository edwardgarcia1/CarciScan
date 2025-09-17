import pandas as pd
import re
import os

def normalize_label(label):
    text = str(label).lower().strip()
    text = re.sub(r"\(.*?\)", "", text).strip()
    if "group 2a" in text or "probably carcinogenic" in text or "2a" in text:
        return "Group 2A"
    elif "group 2b" in text or "possibly carcinogenic" in text or "2b" in text or "class b carcinogen" in text:
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
    # elif "not listed" in text:
    #     return "Not listed by IARC"
    # # Fallback
    else:
        return "DROP"

if __name__ == "__main__":
    dataset = os.path.join(os.path.dirname(__file__), '../../dataset/all_toxin_data_fixed.csv')
    df = pd.read_csv(dataset)
    # Map all labels using normalize_label
    df['normalized'] = df['carcinogenicity'].apply(normalize_label)
    # Count each classification
    counts = df['normalized'].value_counts()
    print("Carcinogenicity classification counts:")
    for label, count in counts.items():
        print(f"  {label}: {count}")
