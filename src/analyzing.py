import numpy as np
import pandas as pd

def extract_feature_importance(model, model_type, top_n=15):
    """
    Extract feature importance / coefficients from a fitted
    LR, RF, or XGBoost pipeline.

    For logistic regression:
        Returns separate tables for positive and negative coefficients.

    For RF/XGBoost:
        Returns one table ranked by feature importance.

    Args:
        model: fitted model pipeline
        model_type (str): "lr", "rf", or "xgb"
        top_n (int): number of features to return

    Returns:
        dict: feature importance tables
    """

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    if model_type == "lr":

        values = estimator.coef_[0]

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": values,
            "odds_ratio": np.exp(values)
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

        return {
            "positive": positive,
            "negative": negative
        }

    elif model_type in ["rf", "xgb"]:

        values = estimator.feature_importances_

        importance_df = pd.DataFrame({
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
            .head(top_n)
            .reset_index(drop=True)
        )

        return {"importance": importance_df}

    else:
        raise ValueError("model_type must be 'lr', 'rf', or 'xgb'")
