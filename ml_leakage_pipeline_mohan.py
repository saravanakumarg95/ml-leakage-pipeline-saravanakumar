from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass
class DepthResult:
    max_depth: int
    train_accuracy: float
    test_accuracy: float

    @property
    def gap(self) -> float:
        return self.train_accuracy - self.test_accuracy


def format_table(rows: list[DepthResult]) -> str:
    header = f"{'max_depth':>10} | {'train_acc':>10} | {'test_acc':>10} | {'gap':>10}"
    divider = "-" * len(header)
    body = [
        f"{r.max_depth:>10} | {r.train_accuracy:>10.4f} | {r.test_accuracy:>10.4f} | {r.gap:>10.4f}"
        for r in rows
    ]
    return "\n".join([header, divider, *body])


def pick_best_depth(rows: list[DepthResult]) -> DepthResult:
    # Prefer higher test accuracy; break ties with lower generalization gap.
    return sorted(rows, key=lambda r: (-r.test_accuracy, r.gap))[0]


def main() -> None:
    X, y = make_classification(n_samples=1000, n_features=10, random_state=RANDOM_STATE)

    print("=== Task 1: Flawed Baseline (Leakage) ===")
    scaler = StandardScaler()
    X_scaled_full = scaler.fit_transform(X)

    X_train_bad, X_test_bad, y_train_bad, y_test_bad = train_test_split(
        X_scaled_full,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    bad_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    bad_model.fit(X_train_bad, y_train_bad)

    bad_train_acc = accuracy_score(y_train_bad, bad_model.predict(X_train_bad))
    bad_test_acc = accuracy_score(y_test_bad, bad_model.predict(X_test_bad))

    print(f"Train accuracy (leaky): {bad_train_acc:.4f}")
    print(f"Test accuracy (leaky):  {bad_test_acc:.4f}")
    print(
        "Issue: StandardScaler was fit on the entire dataset before splitting, "
        "so information from the test set leaked into training preprocessing."
    )

    print("\n=== Task 2: Corrected Pipeline + Cross-Validation ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )

    pipe.fit(X_train, y_train)
    corrected_train_acc = accuracy_score(y_train, pipe.predict(X_train))
    corrected_test_acc = accuracy_score(y_test, pipe.predict(X_test))
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")

    print(f"Train accuracy (pipeline): {corrected_train_acc:.4f}")
    print(f"Test accuracy (pipeline):  {corrected_test_acc:.4f}")
    print(f"5-fold CV mean accuracy:   {np.mean(cv_scores):.4f}")
    print(f"5-fold CV std deviation:   {np.std(cv_scores):.4f}")

    print("\n=== Task 3: Decision Tree Depth Experiment ===")
    depths = [1, 5, 20]
    depth_results: list[DepthResult] = []

    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        tree.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, tree.predict(X_train))
        test_acc = accuracy_score(y_test, tree.predict(X_test))
        depth_results.append(DepthResult(depth, train_acc, test_acc))

    print(format_table(depth_results))

    best = pick_best_depth(depth_results)
    print(
        "\nInterpretation: "
        f"max_depth={best.max_depth} offers the best balance in this run "
        f"(highest test accuracy with a smaller train-test gap)."
    )


if __name__ == "__main__":
    main()
