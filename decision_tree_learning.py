import math
from collections import Counter

examples = [
    {"Outlook": "Sunny", "Wind": "Weak", "PlayTennis": "No"},
    {"Outlook": "Overcast", "Wind": "Strong", "PlayTennis": "Yes"},
    {"Outlook": "Rain", "Wind": "Weak", "PlayTennis": "Yes"},
    {"Outlook": "Sunny", "Wind": "Strong", "PlayTennis": "No"},
    {"Outlook": "Overcast", "Wind": "Weak", "PlayTennis": "Yes"},
    {"Outlook": "Rain", "Wind": "Strong", "PlayTennis": "No"},
    {"Outlook": "Rain", "Wind": "Weak", "PlayTennis": "Yes"},
]
attributes = ["Outlook", "Wind"]
target_attr = "PlayTennis"


def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    counts = Counter(labels)  # frequency dict
    entropy = 0
    for count in counts.values():
        p = count / len(labels)
        entropy += -p * math.log2(p)
    return entropy


def calculate_information_gain(
    examples: list[dict], attr: str, target_attr: str
) -> float:
    """Calculate the information gain of splitting on attr."""
    labels = [example[target_attr] for example in examples]
    parent_entropy = calculate_entropy(labels)

    groups = {}

    for example in examples:
        value = example[attr]
        if value not in groups:
            groups[value] = []
        groups[value].append(example)  # Dict(key->str : value->List[dict])

    weighted_entropy = 0
    for group in groups.values():  # group -> List[dict]
        group_labels = [example[target_attr] for example in group]
        group_entropy = calculate_entropy(group_labels)  # H(S_v)
        weight = len(group) / len(examples)  # S_v / S
        weighted_entropy += weight * group_entropy

    return parent_entropy - weighted_entropy


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class (final class of target_attr) when the tree reaches fallback, else split the tree by IG. Break ties alphabetically."""
    labels = [example[target_attr] for example in examples]
    counts = Counter(labels)

    # if value is equal, sort key alphabetically
    sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

    return sorted_counts[0][0]


def learn_decision_tree(
    examples: list[dict], attributes: list[str], target_attr: str
) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    labels = [example[target_attr] for example in examples]

    if len(set(labels)) == 1:
        return labels[0]

    if not attributes:
        return majority_class(examples, target_attr)

    best_attr = max(
        attributes,
        key=lambda attr: calculate_information_gain(examples, attr, target_attr),
    )

    groups = {}

    for example in examples:
        groups.setdefault(example[best_attr], []).append(example)
    remaining = [attr for attr in attributes if attr != best_attr]

    tree = {best_attr: {}}
    for value in sorted(groups):  # các nhánh con của best_attr
        tree[best_attr][value] = learn_decision_tree(
            groups[value], remaining, target_attr
        )

    return tree


# 1. Custom model (from scratch)
tree_custom = learn_decision_tree(examples, attributes, target_attr)
print("Decision Tree (From Scratch/ID3):")
print(tree_custom)

# 2. Using scikit-learn library
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

df = pd.DataFrame(examples)
X_raw = df[attributes]
y_raw = df[target_attr]

# encode categorical data into int
encoder = OrdinalEncoder()
X = encoder.fit_transform(X_raw)

clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X, y_raw)

print("\nDecision Tree (scikit-learn/CART):")
print(export_text(clf, feature_names=attributes))
