import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def extract_feature_importance(model, model_type, top_n=15, feature_path=None):
    """
    Extract feature importance / coefficients from a fitted
    LR, RF, or XGBoost pipeline. Also saving a csv of feature names for each fitted model

    For logistic regression:
        Returns separate tables for positive and negative coefficients.

    For RF/XGBoost:
        Returns one table ranked by feature importance.

    Args:
        model: fitted model pipeline
        model_type (str): "lr", "rf", or "xgb"
        top_n (int): number of features to return
        feature_path: for storing the feature names

    Returns:
        dict: feature importance tables
    """

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    if model_type == "lr":
        values = estimator.coef_[0]
        importance_df = pd.DataFrame({
            "index": range(len(feature_names)),
            "feature": feature_names,
            "coefficient": values,
            "odds_ratio": np.exp(values),
            "importance": np.abs(values)
        })

        # Clean feature names
        importance_df["feature"] = importance_df["feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
        
        positive = (
            importance_df[importance_df["coefficient"] > 0]
            .sort_values("coefficient", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        negative = (
            importance_df[importance_df["coefficient"] < 0]
            .sort_values("coefficient", ascending=True)
            .head(top_n)
            .reset_index(drop=True)
        )

        result = {
            "positive": positive,
            "negative": negative
        }

    elif model_type in ["rf", "xgb"]:
        values = estimator.feature_importances_
        importance_df = pd.DataFrame({
            "index": range(len(feature_names)),
            "feature": feature_names,
            "importance": values
        })
        importance_df["feature"] = (
            importance_df["feature"]
            .str.replace("num__", "", regex=False)
            .str.replace("cat__", "", regex=False)
        )

        importance_df = (
            importance_df
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        result = {"importance": importance_df}

    else:
        raise ValueError("model_type must be 'lr', 'rf', or 'xgb'")

    # extract the feature names for family mapping downstream
    if feature_path is not None:
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        importance_df.to_csv(feature_path, index=False)
        return result

def assign_feature_family(feature):
    """
    Assign a feature to a predefined feature family.
    Risk features override all other family rules.

    Args:
        feature (str): the feature name

    Returns:
        None type
    """
    feature_lower = feature.lower()
    
    # Risk
    if ("high_oop_burden" in feature_lower or "high_payer_cost" in feature_lower):
        return "Risk"
        
    # Coverage
    if "insur_cvrage_tot_mons" in feature_lower:
        return "Coverage"

    # Cost
    if any(term in feature_lower for term in [
        "cost",
        "reimb_allowed_diff",
        "rejected_amt",
        "claim_processing_burden",
        "total_medoop",
        "cost_adj"
    ]):
        return "Cost"

    # Utilization
    if ("duration" in feature_lower
        or "days_supplied" in feature_lower
        or "days_medsupplied" in feature_lower
        or ("num_" in feature_lower and "_claims" in feature_lower)):
        return "Utilization"

    # Chronic code counts
    if "chronic_code_counts" in feature_lower:
        return "Chronic Count"

    # Disease
    if ("has_" in feature_lower or "esrd" in feature_lower):
        return "Disease"

    # Geography
    if ("bene_county_cd" in feature_lower or "state" in feature_lower):
        return "Geography"

    # Demographics
    if ("race" in feature_lower or "sex" in feature_lower or "age" in feature_lower):
        return "Demographics"
        
    else:
        return "Other"


def run_shap_analysis(model, model_type, year, target, X_test, top_n=3, summary_n=20, output_dir="results/shap"):
    """
    Run SHAP analysis for a fitted XGBoost pipeline.

    The function:
        - transforms the test set using the fitted preprocessor
        - extracts transformed feature names
        - calculates SHAP values
        - ranks features by mean absolute SHAP value
        - saves the SHAP summary plot
        - saves dependence plots for the top features
        - saves the SHAP importance table
        - saves the full SHAP value matrix
        - saves the transformed test set

    Args:
        model: Fitted sklearn Pipeline containing a "preprocessor"
            step and a "model" step.
        model_type (str): Model type. Must be "xgb".
        year (int): Year associated with the model and test set.
        target (str): Prediction target, such as "High_OOP_Burden"
            or "High_Payer_Cost".
        X_test (pd.DataFrame): Original, untransformed test feature
            matrix corresponding to the fitted model.
        top_n (int): Number of top features for which dependence plots
            are generated. Defaults to 3.
        summary_n (int): Number of features displayed in the SHAP
            summary plot. Defaults to 20.
        output_dir (str): Directory in which SHAP results are saved.
            Defaults to "results/shap".

    Returns:
        dict: Dictionary containing:
            - "shap_values": SHAP values for every test observation
              and transformed feature.
            - "X_test_shap": Transformed test feature matrix with
              readable feature names.
            - "shap_importance": DataFrame ranked by mean absolute
              SHAP value.
            - "top_features": List of the top_n most influential
              features.
            - "output_dir": Path to the saved SHAP results.
    """
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    X_test_transformed = preprocessor.transform(X_test)

    # Convert sparse matrix to dense if necessary
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    feature_names = (pd.Series(feature_names)
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .values
    )
    # Create transformed test DataFrame
    X_test_shap = pd.DataFrame(
        X_test_transformed,
        columns=feature_names,
        index=X_test.index
    )
    # Calculate SHAP values
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_test_shap)
    # Some SHAP versions return a list for binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Save model output
    model_output_dir = os.path.join(output_dir, f"{model_type}_{year}_{target}")
    os.makedirs(model_output_dir, exist_ok=True)

    # Calculate SHAP feature importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    })
    shap_importance = shap_importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    top_features = shap_importance.head(top_n)["feature"].tolist()

    # Save SHAP importance table
    shap_importance.to_csv(os.path.join(model_output_dir, "shap_importance.csv"),
        index=False)

    # Save full SHAP value matrix
    shap_values_df = pd.DataFrame(shap_values, columns=feature_names,
        index=X_test.index
    )
    shap_values_df.to_csv(os.path.join(model_output_dir, "shap_values.csv"))

    # Save transformed test set
    X_test_shap.to_csv(os.path.join(model_output_dir, "X_test_transformed.csv"))

    # Save SHAP summary plot
    shap.summary_plot(shap_values, X_test_shap,
                      max_display=summary_n, show=False)
    plt.title(f"SHAP Summary: {model_type.upper()} " f"{year} {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, "summary.png"), dpi=300,
        bbox_inches="tight")
    plt.close("all")

    # Save dependene plots for top features
    for i, feature in enumerate(top_features, start=1):
        shap.dependence_plot(feature, shap_values, X_test_shap, show=False)
        plt.title(
            f"{model_type.upper()} {year} {target}\n"
            f"SHAP Dependence: {feature}"
        )
        plt.tight_layout()
        # santize filename, ran into an error earlier with injury/frailty causing confusions
        safe_feature = re.sub(r'[<>:"/\\|?*]', "_", feature)
        plt.savefig(
            os.path.join(
                model_output_dir,
                f"dependence_{i}_{safe_feature}.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close("all")

    # Return results
    return {"shap_values": shap_values,
            "X_test_shap": X_test_shap,
            "shap_importance": shap_importance,
            "top_features": top_features,
            "output_dir": model_output_dir}





    

      
        
