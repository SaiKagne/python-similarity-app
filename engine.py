# Stages:
#   1. DynamicEBShrinker         → Empirical Bayes shrinkage
#   2. FeatureProcessor          → Invert + RobustScaler + T20 weights
#   3. NumericSimilarity         → Euclidean distance + exponential kernel
#   4. NLPSimilarity             → TF-IDF cosine on keywords
#   5A. CompositeRanker          → Weighted blend + Top N
#   5B. ScoutingReportGenerator  → Z-score normalized NL statements
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity as sk_cosine
from sklearn.feature_extraction.text import TfidfVectorizer


# STAGE 1 — Empirical Bayes Shrinkage
class DynamicEBShrinker:
    """
    Applies Empirical Bayes Shrinkage dynamically based on the filtered role pool.
    Regresses noisy small-sample rates toward the role-pool mean.
    """

    # Each metric mapped to the sample size that determines its reliability
    METRIC_DENOMINATORS = {
        'Avg':              'Outs',    
        'Strike rate':      'Balls',
        'Boundary Runs %':  'Runs',
        'Boundary Rate':    'Balls',
        'Six Runs %':       'Runs',
        'Six Rate':         'Balls',
        'Non-boundary SR':  'Balls',
        'Out Rate':         'Balls',
        'Dots %':           'Balls',
        'Dots Rate':        'Balls'
    }

    def _estimate_M(self, Y: np.ndarray, n: np.ndarray, mu: float) -> float:
        """Estimate shrinkage constant M via Method of Moments."""
        if len(Y) < 3:
            return 300.0  # Safe fallback for tiny pools

        var_total = np.var(Y, ddof=1)
        var_sampling = np.mean(mu / np.maximum(n, 1)) if mu > 0 else 1.0
        var_between = max(var_total - var_sampling, 1e-5)

        M = np.median(n) * (var_sampling / var_between)
        return float(np.clip(M, 50.0, 1000.0))

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assumes df is already filtered by selected roles.
        Returns dataframe with shrunk metric columns overwriting originals.
        """
        shrunk_df = df.copy()

        for metric, denom_col in self.METRIC_DENOMINATORS.items():
            if metric not in df.columns or denom_col not in df.columns:
                continue

            Y = df[metric].fillna(0).values
            n = df[denom_col].fillna(0).values

            # Pool weighted mean (μ)
            mu = np.average(Y, weights=n) if n.sum() > 0 else np.mean(Y)

            # Dynamic prior strength (M)
            M = self._estimate_M(Y, n, mu)

            # Shrinkage weight
            weight = n / (n + M)

            # Apply shrinkage: (weight × observed) + ((1 - weight) × μ)
            shrunk_df[metric] = (weight * Y) + ((1 - weight) * mu)

        return shrunk_df



# STAGE 2 — Feature Processor
class FeatureProcessor:
    """
    Inverts lower-is-better metrics, applies RobustScaler,
    and multiplies by T20 domain weights.
    """

    LOWER_IS_BETTER = ['Dots %', 'Boundary Rate', 'Six Rate']

    DEFAULT_WEIGHTS = {
        "Avg":              1.25,
        "Strike rate":      1.25,
        "Boundary Runs %":  1.20,
        "Boundary Rate":    1.20,
        "Six Runs %":       1.20,
        "Six Rate":         1.20,
        "Non-boundary SR":  1.00,
        "Out Rate":         1.00,
        "Dots %":           1.00,
        "Dots Rate":        1.00
    }

    def __init__(self, custom_weights: dict = None):
        self.weights = custom_weights if custom_weights else self.DEFAULT_WEIGHTS
        self.scaler = RobustScaler()
        self.features = list(self.DEFAULT_WEIGHTS.keys())

    def fit_transform(self, shrunk_df: pd.DataFrame) -> pd.DataFrame:
        """Returns weighted feature matrix ready for distance calculation."""
        processed_df = shrunk_df.copy()

        # Invert "lower is better" metrics so higher is mathematically better
        for metric in self.LOWER_IS_BETTER:
            if metric in processed_df.columns:
                processed_df[metric] = -processed_df[metric]

        # Extract numeric matrix
        X = processed_df[self.features].values

        # RobustScaler (median + IQR) — outlier-resistant
        X_scaled = self.scaler.fit_transform(X)

        # Apply domain weights
        weight_array = np.array([self.weights[f] for f in self.features])
        X_weighted = X_scaled * weight_array

        # Return as DataFrame with Player names attached
        weighted_df = pd.DataFrame(X_weighted, columns=self.features, index=processed_df.index)
        weighted_df['Player'] = processed_df['Player'].values

        return weighted_df


# STAGE 3 — Numeric Similarity (Euclidean + Exponential Kernel)
class NumericSimilarity:
    """
    Computes Euclidean distance between target and all players,
    then converts to similarity via exponential decay kernel.

    Kernel: exp(-sensitivity × distance/median_distance) × 100
    - Anchored to median (robust to outliers)
    - Natural saturation for near-identical players
    - Smooth decay for dissimilar players
    """

    def __init__(self, sensitivity: float = 0.7, bandwidth_floor: float = 0.5):
        self.sensitivity = sensitivity
        self.bandwidth_floor = bandwidth_floor

    def compute(self, processed_df: pd.DataFrame, target_player: str) -> pd.DataFrame:
        feature_cols = [c for c in processed_df.columns if c != 'Player']

        target_mask = processed_df['Player'] == target_player
        if not target_mask.any():
            raise ValueError(f"Target player '{target_player}' not found in processed data.")

        target_vec = processed_df.loc[target_mask, feature_cols].values
        all_vecs = processed_df[feature_cols].values

        # Euclidean distance from target to all players
        distances = euclidean_distances(target_vec, all_vecs)[0]

        # Robust exponential decay kernel
        med_dist = np.median(distances)
        bandwidth = max(med_dist, self.bandwidth_floor)  # prevents score inflation in tight pools
        scores = np.exp(-self.sensitivity * (distances / bandwidth)) * 100.0

        result = pd.DataFrame({
            'Player': processed_df['Player'].values,
            'numeric_score': np.round(scores, 2)
        })

        return result[result['Player'] != target_player].reset_index(drop=True)


# STAGE 4 — NLP Similarity (TF-IDF)
class NLPSimilarity:
    """
    TF-IDF cosine similarity on Strengths and Weaknesses keyword lists.
    Rare shared keywords carry more weight than common ones.
    """

    def __init__(self):
        # tokenizer=lambda x: x tells sklearn "input is already tokenized"
        self.strength_vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None
        )
        self.weakness_vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None
        )

    @staticmethod
    def _parse_keywords(text) -> list:
        if pd.isna(text) or str(text).strip() == '':
            return []
        return [kw.strip().lower() for kw in str(text).split(',') if kw.strip()]

    def compute(
        self,
        df: pd.DataFrame,
        target_player: str,
        user_strengths: list = None,
        user_weaknesses: list = None
    ) -> pd.DataFrame:
        work = df.copy().reset_index(drop=True)

        work['str_list'] = work['Player Strengths'].apply(self._parse_keywords)
        work['wk_list']  = work['Player Weaknesses'].apply(self._parse_keywords)

        target_row = work[work['Player'] == target_player]
        if target_row.empty:
            raise ValueError(f"Target player '{target_player}' not found.")

        # Fallback: use target player's own keywords if user left multiselect empty
        target_str = [s.strip().lower() for s in user_strengths] if user_strengths else target_row.iloc[0]['str_list']
        target_wk  = [w.strip().lower() for w in user_weaknesses] if user_weaknesses else target_row.iloc[0]['wk_list']

        # Strengths TF-IDF
        str_matrix = self.strength_vectorizer.fit_transform(work['str_list'].tolist())
        target_str_vec = self.strength_vectorizer.transform([target_str])
        str_sim = sk_cosine(target_str_vec, str_matrix)[0]

        # Weaknesses TF-IDF
        wk_matrix = self.weakness_vectorizer.fit_transform(work['wk_list'].tolist())
        target_wk_vec = self.weakness_vectorizer.transform([target_wk])
        wk_sim = sk_cosine(target_wk_vec, wk_matrix)[0]

        result = pd.DataFrame({
            'Player': work['Player'].values,
            'strength_score': np.round(str_sim * 100, 2),
            'weakness_score': np.round(wk_sim  * 100, 2)
        })

        return result[result['Player'] != target_player].reset_index(drop=True)


# STAGE 5A — Composite Ranker
class CompositeRanker:
    """
    Blends numeric + NLP scores into a final ranking.
    Default weights: Numeric 0.70 | Strengths 0.20 | Weaknesses 0.10
    """

    DEFAULT_COMPOSITE_WEIGHTS = {
        'numeric':  0.70,
        'strength': 0.20,
        'weakness': 0.10
    }

    def __init__(self, composite_weights: dict = None):
        self.weights = composite_weights if composite_weights else self.DEFAULT_COMPOSITE_WEIGHTS

    def rank(self, results_raw: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        df = results_raw.copy()

        # Normalize weights so they always sum to 1.0 (safe for arbitrary slider values)
        w = self.weights
        total = w['numeric'] + w['strength'] + w['weakness']
        if total == 0:
            wn, ws, ww = 1.0, 0.0, 0.0  # fallback: pure numeric
        else:
            wn, ws, ww = w['numeric']/total, w['strength']/total, w['weakness']/total

        df['overall_similarity'] = np.round(
            wn * df['numeric_score'] +
            ws * df['strength_score'] +
            ww * df['weakness_score'],
            2
        )

        df = df.sort_values('overall_similarity', ascending=False).head(top_n).reset_index(drop=True)
        df.insert(0, 'Rank', range(1, len(df) + 1))

        return df