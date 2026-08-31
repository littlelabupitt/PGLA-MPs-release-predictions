import itertools

import pandas as pd
from sklearn.ensemble import (
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from evaluation import (
    EXTERNAL_TEST_SETS,
    FEATURE_COLS,
    FORMULATION_IDS,
    RANDOM_STATE,
    REVISED_TRAINING_PATH,
    TARGET_COL,
    clip_release,
    evaluate,
    load_external,
    load_training,
)


OUTPUT_SUMMARY = "ensemble_ablation_position_summary.csv"
OUTPUT_FORMULATION = "ensemble_ablation_position_formulation_metrics.csv"


def make_tuned_stack(order=("hgb", "rf", "ridge")):
    estimators = {
        "hgb": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.02,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
        "rf": RandomForestRegressor(
            n_estimators=300,
            min_samples_split=4,
            min_samples_leaf=1,
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "ridge": Ridge(alpha=0.1),
    }
    return StackingRegressor(
        estimators=[(name, estimators[name]) for name in order],
        final_estimator=Ridge(alpha=0.25),
        passthrough=True,
        n_jobs=1,
        cv=3,
    )


def make_bag_model():
    return BaggingRegressor(
        estimator=DecisionTreeRegressor(
            max_depth=8,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        n_estimators=160,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def make_vote_model(order=("hgb", "et", "gbr")):
    estimators = {
        "hgb": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.025,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        ),
        "et": ExtraTreesRegressor(
            n_estimators=260,
            max_depth=16,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.025,
            max_depth=3,
            subsample=0.85,
            random_state=RANDOM_STATE,
        ),
    }
    return VotingRegressor(
        estimators=[(name, estimators[name]) for name in order],
        n_jobs=1,
    )


def make_component(component_name):
    if component_name == "tuned_stack":
        return make_tuned_stack()
    if component_name == "bag":
        return make_bag_model()
    if component_name == "vote":
        return make_vote_model()
    raise ValueError(f"Unknown component: {component_name}")


def weighted_prediction(fitted_components, X, weights):
    pred = 0
    for component_name, weight in weights.items():
        pred = pred + weight * clip_release(fitted_components[component_name].predict(X))
    return clip_release(pred)


def add_metric_rows(rows, label, fitted_components, weights, X_train, X_test, y_train, y_test):
    for dataset_name, X_part, y_part in [
        ("Train", X_train, y_train),
        ("Test", X_test, y_test),
    ]:
        row = {
            "Scenario": label,
            "Dataset": dataset_name,
            "Weights": ";".join(f"{name}={weight:.6f}" for name, weight in weights.items()),
            "Rows": len(y_part),
        }
        row.update(evaluate(y_part, weighted_prediction(fitted_components, X_part, weights)))
        rows.append(row)

    for external_name, path in EXTERNAL_TEST_SETS.items():
        _, X_external, y_external = load_external(path)
        row = {
            "Scenario": label,
            "Dataset": external_name,
            "Weights": ";".join(f"{name}={weight:.6f}" for name, weight in weights.items()),
            "Rows": len(y_external),
        }
        row.update(evaluate(y_external, weighted_prediction(fitted_components, X_external, weights)))
        rows.append(row)


def add_formulation_rows(rows, label, fitted_components, weights, df):
    all_pred = weighted_prediction(fitted_components, df[FEATURE_COLS], weights)
    for formulation_id in FORMULATION_IDS:
        subset = df[df["Formulation Index"].eq(formulation_id)]
        row = {
            "Scenario": label,
            "Formulation Index": formulation_id,
            "Rows": len(subset),
            "Weights": ";".join(f"{name}={weight:.6f}" for name, weight in weights.items()),
        }
        row.update(evaluate(subset[TARGET_COL], all_pred[subset.index]))
        rows.append(row)


def main():
    df, X, y = load_training(REVISED_TRAINING_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    component_names = ["tuned_stack", "bag", "vote"]
    split_components = {
        name: make_component(name).fit(X_train, y_train) for name in component_names
    }
    full_components = {name: make_component(name).fit(X, y) for name in component_names}

    rows = []
    formulation_rows = []

    baseline_weights = {"tuned_stack": 5 / 13, "bag": 2 / 13, "vote": 6 / 13}
    add_metric_rows(
        rows,
        "baseline_all_components",
        split_components,
        baseline_weights,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    add_formulation_rows(
        formulation_rows,
        "baseline_all_components",
        full_components,
        baseline_weights,
        df,
    )

    for component_name in component_names:
        weights = {component_name: 1.0}
        add_metric_rows(
            rows,
            f"only_{component_name}",
            split_components,
            weights,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        add_formulation_rows(formulation_rows, f"only_{component_name}", full_components, weights, df)

    for removed_component in component_names:
        remaining = [name for name in component_names if name != removed_component]
        weights = {name: 1 / len(remaining) for name in remaining}
        add_metric_rows(
            rows,
            f"remove_{removed_component}",
            split_components,
            weights,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        add_formulation_rows(
            formulation_rows,
            f"remove_{removed_component}",
            full_components,
            weights,
            df,
        )

    # Position-swap check: for VotingRegressor and StackingRegressor, estimator order should not
    # materially change the predictions. This verifies that assumption.
    for stack_order in itertools.permutations(["hgb", "rf", "ridge"]):
        stack = make_tuned_stack(stack_order).fit(X_train, y_train)
        temp_components = dict(split_components)
        temp_components["tuned_stack"] = stack
        add_metric_rows(
            rows,
            f"stack_order_{'-'.join(stack_order)}",
            temp_components,
            baseline_weights,
            X_train,
            X_test,
            y_train,
            y_test,
        )

    for vote_order in itertools.permutations(["hgb", "et", "gbr"]):
        vote = make_vote_model(vote_order).fit(X_train, y_train)
        temp_components = dict(split_components)
        temp_components["vote"] = vote
        add_metric_rows(
            rows,
            f"vote_order_{'-'.join(vote_order)}",
            temp_components,
            baseline_weights,
            X_train,
            X_test,
            y_train,
            y_test,
        )

    summary = pd.DataFrame(rows)
    formulation = pd.DataFrame(formulation_rows)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    formulation.to_csv(OUTPUT_FORMULATION, index=False)
    print("=== Ablation and position summary ===")
    print(summary.to_string(index=False))
    print("\n=== Formulation metrics ===")
    print(formulation.to_string(index=False))
    print(f"\nSaved {OUTPUT_SUMMARY}")
    print(f"Saved {OUTPUT_FORMULATION}")


if __name__ == "__main__":
    main()
