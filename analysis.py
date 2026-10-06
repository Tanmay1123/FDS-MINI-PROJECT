"""
FDS Mini Project - Heatwave Intelligence (US, 2019-2023)

All the statistics used by the dashboard live in this file. Every formula is
written out by hand (the way the lab experiments ask for it) instead of calling
ready-made functions like np.mean, np.corrcoef or sklearn models.

    Exp 1  attribute classification        -> ATTRIBUTE_TYPES
    Exp 2  NumPy / Pandas data handling    -> load_data, groupby / filtering helpers
    Exp 3  central tendency & variability  -> mean, median, mode, variance, ...
    Exp 4  correlation coefficient         -> correlation_coefficient, correlation_matrix
    Exp 5  regression imputation           -> simple / multiple linear regression
    Exp 6  normalization & discretization  -> min-max, z-score, decimal scaling, K-means

Run this file directly to print the main insights in the terminal:

    python analysis.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = "heatwave_dataset_us.csv"

NUMERIC_COLS = [
    "Max_Temperature_C",
    "Min_Temperature_C",
    "Mean_Temperature_C",
    "Relative_Humidity_pct",
    "Wind_Speed_mps",
    "Surface_Pressure_kPa",
    "Solar_Radiation_MJm2day",
    "Precipitation_mm",
]

LABELS = {
    "Max_Temperature_C": "Max temperature (°C)",
    "Min_Temperature_C": "Min temperature (°C)",
    "Mean_Temperature_C": "Mean temperature (°C)",
    "Relative_Humidity_pct": "Relative humidity (%)",
    "Wind_Speed_mps": "Wind speed (m/s)",
    "Surface_Pressure_kPa": "Surface pressure (kPa)",
    "Solar_Radiation_MJm2day": "Solar radiation (MJ/m²/day)",
    "Precipitation_mm": "Precipitation (mm)",
    "Latitude": "Latitude (°)",
    "Longitude": "Longitude (°)",
}

SHORT = {
    "Max_Temperature_C": "Max temp",
    "Min_Temperature_C": "Min temp",
    "Mean_Temperature_C": "Mean temp",
    "Relative_Humidity_pct": "Humidity",
    "Wind_Speed_mps": "Wind",
    "Surface_Pressure_kPa": "Pressure",
    "Solar_Radiation_MJm2day": "Solar rad.",
    "Precipitation_mm": "Precip.",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASONS = ["Winter", "Spring", "Summer", "Fall"]
SEVERITY_ORDER = ["Normal", "Warm", "Hot", "Severe"]

# Exp 1: (attribute, categorical/numerical, discrete/continuous, scale, justification)
ATTRIBUTE_TYPES = [
    ("Date", "Numerical", "Discrete", "Interval", "Equally spaced calendar days, no true zero date"),
    ("Year", "Numerical", "Discrete", "Interval", "Equal-unit scale, 'year 0' is an arbitrary reference"),
    ("Month", "Numerical", "Discrete", "Interval", "1-12 with equal spacing, no true zero"),
    ("Month_Name", "Categorical", "Discrete", "Ordinal", "Natural sequence Jan→Dec, differences not measurable"),
    ("Season", "Categorical", "Discrete", "Nominal", "Labels with no quantifiable rank"),
    ("Region", "Categorical", "Discrete", "Nominal", "Named categories, no inherent order"),
    ("Location_Name", "Categorical", "Discrete", "Nominal", "City names, no inherent order"),
    ("Latitude", "Numerical", "Continuous", "Interval", "0° is an arbitrary reference, not 'no latitude'"),
    ("Longitude", "Numerical", "Continuous", "Interval", "Same reasoning as Latitude"),
    ("Max_Temperature_C", "Numerical", "Continuous", "Interval", "0 °C is not 'no temperature'"),
    ("Min_Temperature_C", "Numerical", "Continuous", "Interval", "Same as above"),
    ("Mean_Temperature_C", "Numerical", "Continuous", "Interval", "Same as above"),
    ("Relative_Humidity_pct", "Numerical", "Continuous", "Ratio", "0 % = true absence of humidity"),
    ("Wind_Speed_mps", "Numerical", "Continuous", "Ratio", "0 m/s = true absence of wind"),
    ("Surface_Pressure_kPa", "Numerical", "Continuous", "Ratio", "True zero point, ratios are meaningful"),
    ("Solar_Radiation_MJm2day", "Numerical", "Continuous", "Ratio", "0 = true absence of solar radiation"),
    ("Precipitation_mm", "Numerical", "Continuous", "Ratio", "0 mm = true absence of rainfall"),
    ("Heatwave_Flag", "Categorical", "Discrete", "Nominal", "Binary Yes/No label"),
    ("Heat_Severity_Class", "Categorical", "Discrete", "Ordinal", "Ranked Normal < Warm < Hot < Severe"),
]


# ---------------------------------------------------------------- Exp 2: data handling

def find_data_file(name=DATA_FILE):
    """Look for the CSV next to this script, then in the files/ folder."""
    here = Path(__file__).resolve().parent
    for candidate in (here / name, here / "files" / name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name} not found in {here} or {here / 'files'}")


def load_data(path=None):
    df = pd.read_csv(path or find_data_file())
    df["Date"] = pd.to_datetime(df["Date"])
    df["Is_Heatwave"] = (df["Heatwave_Flag"] == "Yes").astype(int)
    return df


# ---------------------------------------------------------------- Exp 3: central tendency & variability

def mean(values):
    x = np.asarray(values, dtype=float)
    return x.sum() / len(x)


def median(values):
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n % 2 == 0:
        return (x[n // 2 - 1] + x[n // 2]) / 2
    return x[n // 2]


def mode(values):
    """Most frequent value (ties -> the smallest one)."""
    counts = {}
    for v in np.asarray(values).tolist():
        counts[v] = counts.get(v, 0) + 1
    best, best_count = None, 0
    for v in sorted(counts):
        if counts[v] > best_count:
            best, best_count = v, counts[v]
    return best


def variance(values):
    x = np.asarray(values, dtype=float)
    m = mean(x)
    return ((x - m) ** 2).sum() / len(x)


def std_dev(values):
    return math.sqrt(variance(values))


def quartiles(values):
    """Q1 and Q3 as the medians of the lower and upper halves."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    lower = x[: n // 2]
    upper = x[n // 2:] if n % 2 == 0 else x[n // 2 + 1:]
    if len(lower) == 0:
        return x[0], x[0]
    return median(lower), median(upper)


def iqr(values):
    q1, q3 = quartiles(values)
    return q3 - q1


def value_range(values):
    x = np.asarray(values, dtype=float)
    return x.max() - x.min()


def coefficient_of_variation(values):
    m = mean(values)
    return std_dev(values) / m * 100 if m != 0 else float("nan")


def summary_table(df, cols=NUMERIC_COLS):
    """Exp 3 result table: one row per attribute."""
    rows = []
    for c in cols:
        x = df[c].to_numpy()
        q1, q3 = quartiles(x)
        rows.append({
            "Attribute": LABELS.get(c, c),
            "Frequency": len(x),
            "Mean": mean(x),
            "Median": median(x),
            "Mode": mode(x),
            "Std Dev": std_dev(x),
            "Variance": variance(x),
            "Range": value_range(x),
            "Q1": q1,
            "Q3": q3,
            "IQR": q3 - q1,
            "CV %": coefficient_of_variation(x),
        })
    return pd.DataFrame(rows)


def grouped_frequency(values, n_classes=8):
    """Frequency distribution over equal-width class intervals + grouped-data statistics."""
    x = np.asarray(values, dtype=float)
    lo, hi = math.floor(x.min()), math.ceil(x.max())
    h = max(1, math.ceil((hi - lo) / n_classes))
    edges = [lo + i * h for i in range(n_classes + 1)]
    while edges[-1] <= hi:           # make sure the max value has a class
        edges.append(edges[-1] + h)
    while len(edges) > 2 and edges[-2] > hi:
        edges.pop()

    freq = []
    for i in range(len(edges) - 1):
        freq.append(int(((x >= edges[i]) & (x < edges[i + 1])).sum()))

    n = sum(freq)
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(freq))]
    g_mean = sum(f * m for f, m in zip(freq, mids)) / n
    g_var = sum(f * (m - g_mean) ** 2 for f, m in zip(freq, mids)) / n

    def value_at(position):          # interpolate inside the class holding this position
        cum = 0
        for i, f in enumerate(freq):
            if f > 0 and cum + f >= position:
                return edges[i] + (position - cum) / f * h
            cum += f
        return edges[-1]

    k = max(range(len(freq)), key=lambda i: freq[i])      # modal class
    f1 = freq[k]
    f0 = freq[k - 1] if k > 0 else 0
    f2 = freq[k + 1] if k < len(freq) - 1 else 0
    denom = 2 * f1 - f0 - f2
    g_mode = edges[k] + (f1 - f0) / denom * h if denom else mids[k]

    table = pd.DataFrame({
        "Class interval": [f"{edges[i]} – {edges[i + 1]}" for i in range(len(freq))],
        "Midpoint": mids,
        "Frequency": freq,
        "Cumulative": np.cumsum(freq),
    })
    stats = {
        "N": n,
        "Mean": g_mean,
        "Median": value_at(n / 2),
        "Mode": g_mode,
        "Modal class": f"{edges[k]} – {edges[k + 1]}",
        "Variance": g_var,
        "Std Dev": math.sqrt(g_var),
        "IQR": value_at(3 * n / 4) - value_at(n / 4),
    }
    return table, stats, edges


# ---------------------------------------------------------------- Exp 4: correlation

def correlation_coefficient(x, y):
    """Pearson r = Σ(x-x̄)(y-ȳ) / sqrt(Σ(x-x̄)² · Σ(y-ȳ)²)"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x - mean(x)
    dy = y - mean(y)
    denominator = math.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    if denominator == 0:
        return 0.0
    return (dx * dy).sum() / denominator


def correlation_matrix(df, cols):
    n = len(cols)
    r = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r[i, j] = r[j, i] = correlation_coefficient(df[cols[i]], df[cols[j]])
    return pd.DataFrame(r, index=cols, columns=cols)


def describe_r(r):
    strength = abs(r)
    if strength >= 0.7:
        word = "strong"
    elif strength >= 0.4:
        word = "moderate"
    elif strength >= 0.2:
        word = "weak"
    else:
        return "no linear relationship"
    return f"{word} {'positive' if r > 0 else 'negative'}"


# ---------------------------------------------------------------- Exp 5: regression imputation

def simple_linear_regression(x, y):
    """y = w0 + w1·x by least squares."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_bar, y_bar = mean(x), mean(y)
    w1 = ((x - x_bar) * (y - y_bar)).sum() / ((x - x_bar) ** 2).sum()
    w0 = y_bar - w1 * x_bar
    return w0, w1


def multiple_linear_regression(X, y):
    """b = (XᵀX)⁻¹ XᵀY, with a column of ones added for the intercept b0."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(X)), X])
    return np.linalg.inv(X.T @ X) @ (X.T @ y)


def predict_multiple(b, X):
    X = np.asarray(X, dtype=float)
    return b[0] + X @ b[1:]


def regression_scores(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = actual - predicted
    ss_res = (err ** 2).sum()
    ss_tot = ((actual - mean(actual)) ** 2).sum()
    return {
        "MAE": np.abs(err).sum() / len(err),
        "RMSE": math.sqrt(ss_res / len(err)),
        "R²": 1 - ss_res / ss_tot if ss_tot else float("nan"),
    }


def imputation_experiment(df, target, simple_predictor, multi_predictors,
                          missing_pct=10, seed=42):
    """
    The dataset has no missing values, so we hide a random share of `target`,
    predict the hidden values three ways and compare with the true values.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    hidden = np.zeros(n, dtype=bool)
    hidden[rng.choice(n, size=max(1, int(n * missing_pct / 100)), replace=False)] = True

    y = df[target].to_numpy(dtype=float)
    known, missing = ~hidden, hidden

    # baseline: fill with the mean of the known values
    mean_fill = np.full(missing.sum(), mean(y[known]))

    x = df[simple_predictor].to_numpy(dtype=float)
    w0, w1 = simple_linear_regression(x[known], y[known])
    simple_pred = w0 + w1 * x[missing]

    X = df[multi_predictors].to_numpy(dtype=float)
    b = multiple_linear_regression(X[known], y[known])
    multi_pred = predict_multiple(b, X[missing])

    scores = pd.DataFrame({
        "Mean imputation": regression_scores(y[missing], mean_fill),
        "Simple linear regression": regression_scores(y[missing], simple_pred),
        "Multiple linear regression": regression_scores(y[missing], multi_pred),
    }).T

    return {
        "n_missing": int(missing.sum()),
        "actual": y[missing],
        "x_missing": x[missing],
        "x_known": x[known],
        "y_known": y[known],
        "simple": {"w0": w0, "w1": w1, "pred": simple_pred},
        "multi": {"b": b, "pred": multi_pred, "predictors": multi_predictors},
        "scores": scores,
    }


def recommend_predictors(df, target, candidates, min_r=0.2, max_overlap=0.85):
    """
    Decide which attributes to use for predicting `target` with multiple regression.

    Rule 1: a predictor must be related to the target        -> |r| with the target >= min_r
    Rule 2: it must not repeat a predictor already chosen     -> |r| with that one  <  max_overlap
    The candidates are looked at from the strongest to the weakest.
    """
    r_with_target = {c: correlation_coefficient(df[c], df[target]) for c in candidates}
    strongest_first = sorted(candidates, key=lambda c: abs(r_with_target[c]), reverse=True)

    chosen, rows = [], []
    for c in strongest_first:
        r = r_with_target[c]
        decision = "Use"
        if abs(r) < min_r:
            decision = "Skip: too weak"
        else:
            for other in chosen:
                if abs(correlation_coefficient(df[c], df[other])) >= max_overlap:
                    decision = f"Skip: repeats {SHORT[other]}"
                    break
        if decision == "Use":
            chosen.append(c)
        rows.append({"Attribute": LABELS[c], "r with target": r, "Strength": describe_r(r), "Decision": decision})

    return pd.DataFrame(rows), chosen


# ---------------------------------------------------------------- Exp 6: normalization

def min_max_normalize(values, new_min=0.0, new_max=1.0):
    x = np.asarray(values, dtype=float)
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.full(len(x), new_min)
    return (x - lo) / (hi - lo) * (new_max - new_min) + new_min


def z_score_normalize(values):
    x = np.asarray(values, dtype=float)
    s = std_dev(x)
    if s == 0:
        return np.zeros(len(x))
    return (x - mean(x)) / s


def decimal_scaling(values):
    """v' = v / 10^j, j = smallest integer with max(|v'|) < 1."""
    x = np.asarray(values, dtype=float)
    max_abs = np.abs(x).max()
    j = 0
    while max_abs / (10 ** j) >= 1:
        j += 1
    return x / (10 ** j), j


# ---------------------------------------------------------------- Exp 6: K-means discretization

def kmeans(points, k, max_iter=100, seed=42):
    """
    Plain K-means with Euclidean distance.
    points: (n,) or (n, d) array. Returns labels, centroids, WCSS, iterations.
    Clusters are relabelled so that cluster 0 has the smallest first coordinate.
    """
    X = np.asarray(points, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    rng = np.random.default_rng(seed)

    # step 2: random seed points as starting centroids
    centroids = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = np.full(len(X), -1)

    for iteration in range(1, max_iter + 1):
        # step 3: assign every point to the nearest centroid
        dist = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if (new_labels == labels).all():        # stop: assignments did not change
            break
        labels = new_labels
        # step 4: move each centroid to the mean of its cluster
        for c in range(k):
            members = X[labels == c]
            if len(members):
                centroids[c] = members.sum(axis=0) / len(members)

    order = np.argsort(centroids[:, 0])
    rank = np.empty(k, dtype=int)
    rank[order] = np.arange(k)
    labels = rank[labels]
    centroids = centroids[order]

    wcss = float(((X - centroids[labels]) ** 2).sum())
    return labels, centroids, wcss, iteration


def elbow_wcss(points, k_max=8, seed=42):
    return [kmeans(points, k, seed=seed)[2] for k in range(1, k_max + 1)]


def cluster_intervals(values, labels, k):
    """Turn 1-D clusters into bins: the value range each cluster covers."""
    x = np.asarray(values, dtype=float)
    rows = []
    for c in range(k):
        members = x[labels == c]
        if len(members) == 0:
            continue
        rows.append({
            "Bin": f"Bin {c + 1}",
            "From": members.min(),
            "To": members.max(),
            "Centroid": mean(members),
            "Records": len(members),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- insights

def key_insights(df):
    """Plain-language findings computed from the (filtered) data."""
    out = []
    if len(df) == 0:
        return out

    hw = df[df["Is_Heatwave"] == 1]
    share = len(hw) / len(df) * 100
    out.append(("Heatwaves are rare",
                f"{len(hw):,} of {len(df):,} city-days ({share:.1f}%) are flagged as a heatwave."))

    if len(hw):
        by_region = hw.groupby("Region").size().sort_values(ascending=False)
        top = by_region.index[0]
        out.append(("One region carries most of the risk",
                    f"{top} accounts for {by_region.iloc[0] / len(hw) * 100:.0f}% of all heatwave days "
                    f"({by_region.iloc[0]:,})."))

        by_city = hw.groupby("Location_Name").size().sort_values(ascending=False)
        n_cities = df["Location_Name"].nunique()
        out.append(("A handful of cities dominate",
                    f"{by_city.index[0]} has the most heatwave days ({by_city.iloc[0]:,}); "
                    f"{n_cities - len(by_city)} of {n_cities} cities never had one."))

        by_month = hw.groupby("Month").size()
        peak = by_month.idxmax()
        jja = by_month.reindex([6, 7, 8]).fillna(0).sum()
        out.append(("Heatwaves are a summer event",
                    f"{MONTHS[peak - 1]} is the peak month; June-August hold "
                    f"{jja / len(hw) * 100:.0f}% of heatwave days."))

        by_year = hw.groupby("Year").size()
        if len(by_year) > 1:
            first, last = by_year.index.min(), by_year.index.max()
            change = (by_year[last] - by_year[first]) / by_year[first] * 100
            out.append(("Heatwave days are increasing" if change > 0 else "Heatwave days are decreasing",
                        f"{by_year[first]:,} heatwave days in {first} vs {by_year[last]:,} in {last} "
                        f"({change:+.0f}%)."))

        normal = df[df["Is_Heatwave"] == 0]
        if len(normal):
            out.append(("Heatwave days are dry and sunny",
                        f"Average humidity is {mean(hw['Relative_Humidity_pct']):.0f}% on heatwave days vs "
                        f"{mean(normal['Relative_Humidity_pct']):.0f}% otherwise; solar radiation is "
                        f"{mean(hw['Solar_Radiation_MJm2day']):.1f} vs "
                        f"{mean(normal['Solar_Radiation_MJm2day']):.1f} MJ/m²/day."))

    r_solar = correlation_coefficient(df["Max_Temperature_C"], df["Solar_Radiation_MJm2day"])
    r_hum = correlation_coefficient(df["Max_Temperature_C"], df["Relative_Humidity_pct"])
    out.append(("Sun up, humidity down",
                f"Max temperature has a {describe_r(r_solar)} correlation with solar radiation "
                f"(r = {r_solar:.2f}) and a {describe_r(r_hum)} one with humidity (r = {r_hum:.2f})."))

    hottest = df.loc[df["Max_Temperature_C"].idxmax()]
    out.append(("Hottest reading",
                f"{hottest['Max_Temperature_C']:.1f} °C in {hottest['Location_Name']} on "
                f"{hottest['Date']:%d %b %Y}."))
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 20)
    data = load_data()
    print(f"Loaded {data.shape[0]:,} rows x {data.shape[1]} columns\n")

    print("Central tendency & variability (Exp 3)")
    print(summary_table(data).round(2).to_string(index=False), "\n")

    print("Correlation with Max_Temperature_C (Exp 4)")
    for col in NUMERIC_COLS[1:] + ["Latitude"]:
        r = correlation_coefficient(data["Max_Temperature_C"], data[col])
        print(f"  {col:<26} r = {r:+.3f}  ({describe_r(r)})")
    print()

    print("Regression imputation of Max_Temperature_C, 10% hidden (Exp 5)")
    result = imputation_experiment(
        data, "Max_Temperature_C", "Min_Temperature_C",
        ["Min_Temperature_C", "Solar_Radiation_MJm2day", "Relative_Humidity_pct"])
    print(result["scores"].round(3).to_string(), "\n")

    print("K-means discretization of Max_Temperature_C, k = 4 (Exp 6)")
    lab, cen, wcss, iters = kmeans(data["Max_Temperature_C"].to_numpy(), 4)
    print(cluster_intervals(data["Max_Temperature_C"], lab, 4).round(2).to_string(index=False))
    print(f"  converged in {iters} iterations, WCSS = {wcss:,.0f}\n")

    print("Key insights")
    for title, text in key_insights(data):
        print(f"  - {title}: {text}")
