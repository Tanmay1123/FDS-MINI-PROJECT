"""
FDS Mini Project - Heatwave Intelligence Dashboard

Run with:   streamlit run app.py

analysis.py  -> the statistics (formulas written by hand, as in the lab experiments)
charts.py    -> the Matplotlib figures
style.css    -> how the page looks
app.py       -> this file: it puts everything on the page, top to bottom
"""

import inspect
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import analysis as an
import charts as ch

st.set_page_config(page_title="Heatwave Intelligence", page_icon="🌡️", layout="wide")


# ================================================================ 1. THE LOOK

# One fixed theme. The page colours are in style.css, the chart colours are in charts.py.
css = Path(__file__).with_name("style.css").read_text()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

PURPLE, CORAL, TEAL = ch.PURPLE, ch.CORAL, ch.TEAL
ACCENTS = [PURPLE, CORAL, TEAL]                      # the three accent colours, used in turn


# ================================================================ 2. SMALL HELPERS

def html(text):
    st.markdown(text, unsafe_allow_html=True)


def tiles_html(items):
    """Coloured numbers with a thin underline. items = [(label, value, note), ...]"""
    stats = ""
    for i, (label, value, note) in enumerate(items):
        colour = ACCENTS[i % 3]
        stats += (f'<div><div class="stat-value" style="color:{colour}">{value}</div>'
                  f'<div class="stat-label">{label}</div><div class="stat-note">{note}</div></div>')
    return f'<div class="stats">{stats}</div>'


def hero(eyebrow, title, text, tiles=None):
    """The heading at the top of a page."""
    html(f'<div class="page-head"><div class="eyebrow">{eyebrow}</div>'
         f'<div class="page-title">{title}</div><div class="page-text">{text}</div>'
         f'{tiles_html(tiles) if tiles else ""}</div>')


def in_words(number):
    """37 -> 'Thirty-Seven' (for whole numbers from 0 to 100)."""
    ones = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
            "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
    if number < 20:
        return ones[number]
    if number == 100:
        return "One Hundred"
    if number % 10 == 0:
        return tens[number // 10]
    return tens[number // 10] + "-" + ones[number % 10]


def bars_html(values):
    """Horizontal gradient bars. `values` is a Series: label -> number."""
    colours = [TEAL, CORAL, PURPLE, ch.ORCHID, ch.PEACH]
    rows = ""
    for i, (label, value) in enumerate(values.items()):
        width = value / values.max() * 100
        rows += (f'<div class="bar-row"><div class="bar-track">'
                 f'<div class="bar" style="width:{width:.0f}%; --c:{colours[i % 5]}">{label}</div></div>'
                 f'<div class="bar-value">{value:,}</div></div>')
    return rows


def ranking_html(values):
    """The top three in large coloured type, the rest as a small list."""
    total = values.sum()
    rows = ""
    for i, (name, value) in enumerate(values.items()):
        share = f"{value:,} ({value / total * 100:.1f}%)"
        if i < 3:
            colour = [TEAL, CORAL, PURPLE][i]
            rows += (f'<div class="rank-big" style="color:{colour}"><span class="rank-name">{name}</span>'
                     f'<span class="rank-value">{share}</span></div>')
        else:
            rows += f'<div class="rank-small"><span>{name}</span><span>{share}</span></div>'
    return rows


def section(title):
    html(f'<div class="section"><div class="section-title">{title}</div></div>')


def reading(text):
    html(f'<div class="reading"><b>Reading it:</b> {text}</div>')


def card_name(title):
    """A name for the card that is safe to use in CSS, e.g. 'overview-heatwave-days-by-year'."""
    return "".join(c if c.isalnum() else "-" for c in f"{page} {title}".lower())


def card_title(title, note):
    html(f'<div class="card-title">{title}</div><div class="card-note">{note}</div>')


def card(title, note=""):
    """A plain card for tables and text. Use it as:  with card("Title"): ..."""
    box = st.container(border=True, key="card-" + card_name(title))
    with box:
        card_title(title, note)
    return box


@st.fragment          # only this card is redrawn when its Code switch is used
def chart_card(title, plot, *args, note="", read="", maths=None, **kwargs):
    """
    A card with one chart. `plot` is a function from charts.py and *args are its inputs.
    The Code switch flips the card to show the code of `plot` (and of `maths`, if given).
    """
    name = card_name(title)
    show_code = st.session_state.get("code-" + name, False)

    with st.container(border=True, key=f"card-{name}{'--flip' if show_code else ''}"):
        with st.container(horizontal=True, horizontal_alignment="distribute", key="head-" + name):
            card_title(title, note)
            st.toggle("Code", key="code-" + name)

        if show_code:
            st.caption("The code that draws this chart")
            st.code(inspect.getsource(plot), language="python")
            if maths:
                st.caption("analysis.py — the formula behind it")
                st.code(inspect.getsource(maths), language="python")
        else:
            figure = plot(*args, **kwargs)
            if figure is not None:          # a Matplotlib figure; live charts draw themselves
                st.pyplot(figure)
                plt.close(figure)
            if read:
                reading(read)


# results that take a moment to compute are remembered with st.cache_data

@st.cache_data
def load_data():
    return an.load_data()


@st.cache_data
def summary_table(df):
    return an.summary_table(df)


@st.cache_data
def kmeans(values, k):
    return an.kmeans(values, k)


@st.cache_data
def elbow(values):
    return an.elbow_wcss(values, 8)


def to_excel(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Data", index=False)
        an.summary_table(df).round(3).to_excel(writer, sheet_name="Summary statistics", index=False)
    return buffer.getvalue()


# Live charts for the overview. These use the charts built into Streamlit (st.area_chart and
# st.bar_chart), so you can hover to read values, drag to move and scroll to zoom.

def live_hot_days(df):
    """Stacked area chart: Warm, Hot and Severe days in every month."""
    hot = df[df["Heat_Severity_Class"] != "Normal"]
    month_start = hot["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = pd.crosstab(month_start, hot["Heat_Severity_Class"])        # rows = months, columns = classes
    monthly = monthly.reindex(columns=["Warm", "Hot", "Severe"], fill_value=0)
    st.area_chart(monthly, color=[TEAL, CORAL, PURPLE], stack=True, height=340,
                  x_label="Month", y_label="City-days per month")


def live_days_by_year(by_year):
    """Bar chart: heatwave days in each year."""
    table = pd.DataFrame({"Year": by_year.index.astype(str), "Heatwave days": by_year.to_numpy()})
    st.bar_chart(table, x="Year", y="Heatwave days", color=TEAL, height=340)


def live_region_share(df):
    """Stacked horizontal bars: share of each region's days that were Warm, Hot or Severe."""
    share = pd.crosstab(df["Region"], df["Heat_Severity_Class"], normalize="index") * 100
    share = share.reindex(columns=["Warm", "Hot", "Severe"], fill_value=0).round(1)
    st.bar_chart(share, color=[TEAL, CORAL, PURPLE], horizontal=True, stack=True, height=340,
                 x_label="", y_label="Share of days at 35 °C or above (%)")   # the labels swap when horizontal


# ================================================================ 3. SIDEBAR

data = load_data()

PAGES = ["Overview", "Dataset & attributes", "Central tendency", "Correlation",
         "Regression imputation", "Normalization & K-means", "Plot gallery", "Visualization techniques"]

with st.sidebar:
    html('<div class="brand-name">Heatwave<br>Intelligence</div>'
         '<div class="brand-sub">FDS mini project · 2019–2023</div>')

    html('<div class="side-label">Sections</div>')
    page = st.radio("Section", PAGES, label_visibility="collapsed", width="stretch")

    html('<div class="side-label">Filters</div>')
    years = st.multiselect("Year", sorted(data["Year"].unique()), placeholder="All years")
    regions = st.multiselect("Region", ch.REGIONS, placeholder="All regions")
    seasons = st.multiselect("Season", an.SEASONS, placeholder="All seasons")

# apply the filters: keep only the rows that match what was chosen
df = data
if years:
    df = df[df["Year"].isin(years)]
if regions:
    df = df[df["Region"].isin(regions)]
if seasons:
    df = df[df["Season"].isin(seasons)]

heatwaves = df[df["Is_Heatwave"] == 1]               # only the heatwave days
export = df.drop(columns="Is_Heatwave")              # what the download buttons save
export = export.assign(Date=export["Date"].dt.date)  # dates without a time part

with st.sidebar:
    st.caption(f"{len(df):,} of {len(data):,} records selected")

    html('<div class="side-label">Download data</div>')
    left, right = st.columns(2)
    left.download_button("CSV", lambda: export.to_csv(index=False).encode(), "heatwave_data.csv",
                         "text/csv", width="stretch", on_click="ignore")
    right.download_button("Excel", lambda: to_excel(export), "heatwave_data.xlsx",
                          width="stretch", on_click="ignore")

if len(df) < 50:
    st.warning("These filters leave too few records. Remove a filter to continue.")
    st.stop()


# ================================================================ 4. OVERVIEW

if page == "Overview":
    hottest = df.loc[df["Max_Temperature_C"].idxmax()]
    by_city = heatwaves.groupby("Location_Name").size().sort_values(ascending=False)
    by_region = heatwaves.groupby("Region").size().sort_values(ascending=False)
    by_month = heatwaves.groupby("Month_Name").size().sort_values(ascending=False)
    by_year = heatwaves.groupby("Year").size().reindex(sorted(df["Year"].unique()), fill_value=0)

    first, last = df["Year"].min(), df["Year"].max()
    period = str(first) if first == last else f"{first}–{str(last)[2:]}"

    # top row: the big title on the left, the main chart on the right
    left, right = st.columns([1, 2.1])
    with left:
        html(f'<div class="intro"><div class="intro-year">{period}</div>'
             f'<div class="intro-sub">Heatwave Analytics</div>'
             f'<div class="pill">{df["Location_Name"].nunique()} US cities · {df["Region"].nunique()} regions</div>'
             + tiles_html([
                 ("Heatwave days", f"{len(heatwaves):,}", "max temp ≥ 40 °C"),
                 ("Cities affected", len(by_city), f"of {df['Location_Name'].nunique()}"),
                 ("Hottest reading", f"{hottest['Max_Temperature_C']:.1f} °C", hottest["Location_Name"]),
             ]) + '</div>')
    with right:
        chart_card("Hot Days Per Month", live_hot_days, df,
                   note="City-days at 35 °C or above. Hover over the chart to read any month.")

    # second row: three panels
    one, two, three = st.columns(3)
    with one:
        chart_card("Heatwave Sources", ch.pie, by_region, note="Share of all heatwave days by region.")
    with two:
        with card("When They Strike"):
            if len(heatwaves):
                summer = heatwaves["Month"].isin([6, 7, 8]).mean() * 100
                html(f'<div class="big-word">{in_words(round(summer))}</div>'
                     f'<div class="big-text">Percent of all heatwave days fall in June, July and August.</div>'
                     + bars_html(by_month.head(5)))
            else:
                st.write("No heatwave days in this selection.")
    with three:
        with card("Heatwaves by City"):
            if len(heatwaves):
                html(ranking_html(by_city.head(10)))
            else:
                st.write("No heatwave days in this selection.")

    section("Key Insights")
    insights = ""
    for number, (title, text) in enumerate(an.key_insights(df), start=1):
        colour = ACCENTS[number % 3]
        insights += (f'<div class="insight"><div class="insight-number" style="color:{colour}">{number:02d}</div>'
                     f'<div class="insight-title">{title}</div><div class="insight-text">{text}</div></div>')
    html(f'<div class="insights">{insights}</div>')

    section("Year by Year, Region by Region")
    left, right = st.columns(2)
    with left:
        chart_card("Heatwave Days by Year", live_days_by_year, by_year, note="Hover over a bar to read its value.")
    with right:
        chart_card("How Often Each Region Gets Hot", live_region_share, df,
                   note="Warm = 35–40 °C, Hot = 40–45 °C, Severe = above 45 °C.")


# ================================================================ 5. EXPERIMENTS 1 & 2

elif page == "Dataset & attributes":
    hero("Experiments 1 & 2", "Dataset & attributes",
         "What the dataset contains, how every attribute is classified, and the basic Pandas "
         "commands used to explore it.",
         tiles=[
             ("Rows", f"{df.shape[0]:,}", "df.shape[0]"),
             ("Columns", export.shape[1], "df.shape[1]"),
             ("Missing values", int(df.isnull().sum().sum()), "df.isnull().sum()"),
             ("Years covered", f"{df['Year'].min()} – {df['Year'].max()}", "daily records"),
         ])

    attributes = pd.DataFrame(an.ATTRIBUTE_TYPES, columns=[
        "Attribute", "Categorical / Numerical", "Discrete / Continuous", "Scale", "Justification"])

    section("Experiment 1 · Attribute types")
    with card("Attribute classification", "Every column classified by type and by measurement scale."):
        st.dataframe(attributes, hide_index=True, height=35 * len(attributes) + 38)

    left, right = st.columns(2)
    with left:
        scale_counts = attributes["Scale"].value_counts().reindex(["Nominal", "Ordinal", "Interval", "Ratio"])
        chart_card("Attributes per measurement scale", ch.columns, scale_counts, "Number of attributes")
    with right:
        class_counts = df["Heat_Severity_Class"].value_counts().reindex(an.SEVERITY_ORDER, fill_value=0)
        chart_card("Heat_Severity_Class.value_counts()", ch.hbar, class_counts, "Number of records",
                   note="An ordinal attribute: Normal < Warm < Hot < Severe.")

    section("Experiment 2 · Pandas commands")
    with card("df.head(10)", "The first ten rows."):
        st.dataframe(export.head(10), hide_index=True)

    with card("df.describe()", "Summary statistics of the numeric columns."):
        st.dataframe(df[an.NUMERIC_COLS].describe().T.round(2))

    left, right = st.columns(2)
    with left:
        with card("df[df['Max_Temperature_C'] > 45]", "Filtering: keep only the rows that pass a condition."):
            hot_days = export[export["Max_Temperature_C"] > 45]
            st.write(f"**{len(hot_days):,}** records are above 45 °C.")
            st.dataframe(hot_days[["Date", "Location_Name", "Region", "Max_Temperature_C"]]
                         .sort_values("Max_Temperature_C", ascending=False).head(50),
                         hide_index=True, height=283)
    with right:
        with card("df.groupby('Region').agg(...)", "Aggregation: one summary row per group."):
            grouped = df.groupby("Region").agg(
                Records=("Max_Temperature_C", "size"),
                Mean_max_temp=("Max_Temperature_C", "mean"),
                Highest_max_temp=("Max_Temperature_C", "max"),
                Heatwave_days=("Is_Heatwave", "sum"),
            ).round(2).sort_values("Heatwave_days", ascending=False)
            st.dataframe(grouped)


# ================================================================ 6. EXPERIMENT 3

elif page == "Central tendency":
    hero("Experiment 3", "Central tendency & variability",
         "Mean, median, mode, variance, standard deviation and interquartile range, each computed "
         "with our own function instead of a ready-made one.")

    table = summary_table(df)
    with card("All numeric attributes", "Computed by analysis.py, one row per attribute."):
        st.dataframe(table.round(2), hide_index=True)

    with card("Check: our functions against NumPy", "The hand-written results match NumPy's built-in ones."):
        check = pd.DataFrame({
            "Attribute": table["Attribute"],
            "Our mean": table["Mean"],
            "np.mean": [np.mean(df[c]) for c in an.NUMERIC_COLS],
            "Our median": table["Median"],
            "np.median": [np.median(df[c]) for c in an.NUMERIC_COLS],
            "Our std dev": table["Std Dev"],
            "np.std": [np.std(df[c]) for c in an.NUMERIC_COLS],
        })
        st.dataframe(check.round(4), hide_index=True)

    section("One attribute in detail")
    column = st.selectbox("Attribute", an.NUMERIC_COLS, format_func=an.LABELS.get)
    values = df[column].to_numpy()
    row = table[table["Attribute"] == an.LABELS[column]].iloc[0]

    html(tiles_html([
        ("Mean", f"{row['Mean']:.2f}", "sum ÷ count"),
        ("Median", f"{row['Median']:.2f}", "middle value"),
        ("Mode", f"{row['Mode']:.2f}", "most frequent"),
        ("Std deviation", f"{row['Std Dev']:.2f}", "typical distance from mean"),
        ("Variance", f"{row['Variance']:.2f}", "std deviation squared"),
        ("IQR", f"{row['IQR']:.2f}", "Q3 − Q1"),
    ]))

    if row["Mean"] < row["Median"]:
        skew = "the mean is below the median, so the distribution has a longer tail on the low side."
    else:
        skew = "the mean is above the median, so a few large values pull the mean up."

    left, right = st.columns(2)
    with left:
        markers = {"Mean": (row["Mean"], ch.CORAL), "Median": (row["Median"], ch.INK), "Mode": (row["Mode"], ch.PURPLE)}
        chart_card("Distribution with mean, median and mode", ch.histogram, values, an.LABELS[column],
                   markers=markers, read=skew, maths=an.variance)
    with right:
        frequency_table, grouped, edges = an.grouped_frequency(values, 8)
        chart_card("Grouped data: frequency distribution", ch.frequency_bars, frequency_table,
                   an.LABELS[column], note=f"Modal class: {grouped['Modal class']}", maths=an.grouped_frequency)

    left, right = st.columns(2)
    with left:
        with card("Frequency table", "Eight equal-width class intervals."):
            st.dataframe(frequency_table, hide_index=True)
    with right:
        with card("Grouped vs raw statistics", "Grouping loses a little detail, so the two are close, not equal."):
            measures = ["Mean", "Median", "Mode", "Variance", "Std Dev", "IQR"]
            compare = pd.DataFrame({"Measure": measures,
                                    "From grouped data": [grouped[m] for m in measures],
                                    "From raw data": [row[m] for m in measures]})
            st.dataframe(compare.round(2), hide_index=True)

    left, right = st.columns([3, 2])
    with left:
        chart_card("Spread in every region", ch.box_by_group, df, column, "Region", ch.REGIONS,
                   read="the box is the middle 50% of days, the line inside it is the median, and the "
                        "whiskers show how far the remaining days reach.")
    with right:
        with card("The same comparison in numbers", "df.groupby('Region')[column].agg([...])"):
            by_region = df.groupby("Region")[column].agg(["mean", "median", "std", "min", "max"])
            st.dataframe(by_region.round(2))


# ================================================================ 7. EXPERIMENT 4

elif page == "Correlation":
    hero("Experiment 4", "Correlation",
         "Pearson's correlation coefficient r measures how strongly two attributes move together "
         "in a straight line. It is computed from the formula, without an in-built function.")

    columns = an.NUMERIC_COLS + ["Latitude", "Longitude"]
    matrix = an.correlation_matrix(df, columns)
    r_solar = matrix.loc["Max_Temperature_C", "Solar_Radiation_MJm2day"]
    r_humidity = matrix.loc["Max_Temperature_C", "Relative_Humidity_pct"]
    r_wind = matrix.loc["Wind_Speed_mps", "Precipitation_mm"]

    html(tiles_html([
        ("Temperature & sunshine", f"{r_solar:+.2f}", an.describe_r(r_solar)),
        ("Temperature & humidity", f"{r_humidity:+.2f}", an.describe_r(r_humidity)),
        ("Wind & rainfall", f"{r_wind:+.2f}", an.describe_r(r_wind)),
    ]))

    left, right = st.columns([3, 2])
    with left:
        chart_card("Correlation matrix", ch.corr_heatmap, matrix, maths=an.correlation_coefficient,
                   note="Red = move together, blue = move in opposite directions, grey = unrelated.")
    with right:
        chart_card("What moves with max temperature", ch.corr_bars,
                   matrix["Max_Temperature_C"].drop("Max_Temperature_C"), "max temperature",
                   maths=an.correlation_coefficient)
        with card("The formula"):
            st.latex(r"r=\frac{\sum (x_i-\bar{x})(y_i-\bar{y})}"
                     r"{\sqrt{\sum (x_i-\bar{x})^2\,\sum (y_i-\bar{y})^2}}")
            st.caption("r = +1 perfect positive, r = −1 perfect negative, r = 0 no linear relationship.")

    examples = [("Positive", "Max_Temperature_C", "Solar_Radiation_MJm2day"),
                ("Negative", "Latitude", "Min_Temperature_C"),
                ("None", "Wind_Speed_mps", "Precipitation_mm")]
    chart_card("Positive, negative and no correlation", ch.correlation_examples, df, examples,
               note="The three cases from the experiment, drawn on a sample of 1,500 points.",
               read="hotter days get more sunshine; cities further north have colder nights; wind "
                    "speed tells us nothing about rainfall.")


# ================================================================ 8. EXPERIMENT 5

elif page == "Regression imputation":
    hero("Experiment 5", "Predicting missing values with regression",
         "The dataset has no missing values, so we hide 10% of the maximum temperatures, predict "
         "them with our own least-squares regression, and compare with the true values.")

    target = "Max_Temperature_C"
    one_predictor = "Solar_Radiation_MJm2day"
    many_predictors = ["Solar_Radiation_MJm2day", "Relative_Humidity_pct", "Latitude", "Wind_Speed_mps"]

    result = an.imputation_experiment(df, target, one_predictor, many_predictors, missing_pct=10)
    scores = result["scores"]
    w0, w1 = result["simple"]["w0"], result["simple"]["w1"]
    b = result["multi"]["b"]

    html(tiles_html([
        ("Values hidden", f"{result['n_missing']:,}", "10% of the records"),
        ("Mean imputation", f"{scores.loc['Mean imputation', 'RMSE']:.2f} °C", "error (RMSE)"),
        ("Simple regression", f"{scores.loc['Simple linear regression', 'RMSE']:.2f} °C", "error (RMSE)"),
        ("Multiple regression", f"{scores.loc['Multiple linear regression', 'RMSE']:.2f} °C", "error (RMSE)"),
    ]))

    section("Simple linear regression · one predictor")
    left, right = st.columns([3, 2])
    with left:
        chart_card("The fitted line", ch.scatter_fit, result["x_known"][::30], result["y_known"][::30],
                   an.LABELS[one_predictor], an.LABELS[target], w0, w1, maths=an.simple_linear_regression,
                   note="y = w₀ + w₁·x, fitted by least squares on the known values (every 30th point drawn).")
    with right:
        with card("The formulas"):
            st.latex(r"w_1=\frac{\sum (x_i-\bar x)(y_i-\bar y)}{\sum (x_i-\bar x)^2}=" + f"{w1:.3f}")
            st.latex(r"w_0=\bar y-w_1\bar x=" + f"{w0:.3f}")
            st.write(f"**Max temp = {w0:.2f} + {w1:.3f} × Solar radiation**")

    section("Multiple linear regression · four predictors")
    left, right = st.columns([2, 3])
    with left:
        with card("The coefficients", "b = (XᵀX)⁻¹ XᵀY, solved with matrix algebra."):
            coefficients = pd.DataFrame({"Term": ["Intercept (b₀)"] + [an.LABELS[c] for c in many_predictors],
                                         "Coefficient": b})
            st.dataframe(coefficients.round(4), hide_index=True)
    with right:
        chart_card("Error of each method", ch.error_bars, scores, "RMSE", maths=an.multiple_linear_regression)

    section("How good are the predictions?")
    chart_card("Predicted vs actual for the hidden values", ch.actual_vs_predicted, result, "max temperature (°C)",
               note="Points on the diagonal are perfect predictions; a tighter cloud is a better model.",
               maths=an.imputation_experiment)

    with card("Accuracy of each method", "MAE and RMSE: lower is better. R²: closer to 1 is better."):
        st.dataframe(scores.round(3))
        best = scores["RMSE"].idxmin()
        gain = (1 - scores.loc[best, "RMSE"] / scores.loc["Mean imputation", "RMSE"]) * 100
        reading(f"{best.lower()} is the most accurate. Its error is {gain:.0f}% lower than filling "
                f"every gap with the mean.")


# ================================================================ 9. EXPERIMENT 6

elif page == "Normalization & K-means":
    hero("Experiment 6", "Normalization & discretization",
         "Three ways of putting attributes on a common scale, and K-means clustering used to turn "
         "a continuous attribute into a small number of bins.")

    section("Normalization")
    chart_card("Why normalize?", ch.scale_comparison, df, an.NUMERIC_COLS, maths=an.z_score_normalize,
               note="On raw scales pressure (≈ 95) dwarfs wind speed (≈ 2). After normalization every attribute carries equal weight.")

    column = st.selectbox("Attribute", an.NUMERIC_COLS, format_func=an.LABELS.get)
    values = df[column].to_numpy()
    min_max = an.min_max_normalize(values)
    z_score = an.z_score_normalize(values)
    decimal, j = an.decimal_scaling(values)

    with card("All three techniques side by side", "The first 200 records."):
        consolidated = pd.DataFrame({"City": df["Location_Name"].to_numpy(), "Original": values,
                                     "Min-max": min_max, "Z-score": z_score, "Decimal scaling": decimal})
        st.dataframe(consolidated.head(200).round(4), hide_index=True, height=283)
        left, middle, right = st.columns(3)
        left.latex(r"v'=\frac{v-\min}{\max-\min}")
        middle.latex(r"v'=\frac{v-\bar{x}}{\sigma}")
        right.latex(r"v'=\frac{v}{10^{" + str(j) + "}}")

    versions = {"Min-max": min_max, "Z-score": z_score, f"Decimal scaling (j = {j})": decimal}
    chart_card("Same shape, different scale", ch.normalization_histograms, values, versions, an.LABELS[column],
               maths=an.min_max_normalize,
               read="normalization only moves and stretches the axis. The shape of the distribution does not change.")

    section("K-means discretization")
    k = st.slider("Number of bins (k)", 2, 8, 4)
    temperatures = df["Max_Temperature_C"].to_numpy()
    labels, centroids, wcss, iterations = kmeans(temperatures, k)

    left, right = st.columns([2, 3])
    with left:
        chart_card("Elbow method", ch.elbow, elbow(temperatures), k, maths=an.elbow_wcss,
                   note="The bend in the curve suggests a good k.")
    with right:
        chart_card("Bins found by K-means", ch.kmeans_bins, temperatures, labels, centroids,
                   an.LABELS["Max_Temperature_C"], maths=an.kmeans,
                   note=f"Converged in {iterations} iterations.")

    left, right = st.columns(2)
    with left:
        with card("The bins", "Each temperature is replaced by the bin it falls in."):
            st.dataframe(an.cluster_intervals(temperatures, labels, k).round(2), hide_index=True)
    with right:
        with card("K-means bins vs Heat_Severity_Class"):
            bins = pd.Series([f"Bin {c + 1}" for c in labels], name="K-means bin")
            cross = pd.crosstab(bins, df["Heat_Severity_Class"].to_numpy(), colnames=["Severity class"])
            st.dataframe(cross.reindex(columns=[c for c in an.SEVERITY_ORDER if c in cross.columns]))
            reading("the severity classes are fixed thresholds at 35 / 40 / 45 °C. K-means has no "
                    "thresholds; it finds the natural groups across the whole range.")


# ================================================================ 10. EXPERIMENT 7

elif page == "Plot gallery":
    hero("Experiment 7", "Plot gallery",
         "The six basic Matplotlib plots, each used for the question it answers best.")

    by_year = heatwaves.groupby("Year").size().reindex(sorted(df["Year"].unique()), fill_value=0)
    by_region = heatwaves.groupby("Region").size()
    threshold = {"Heatwave threshold": (40.0, ch.CORAL)}

    left, right = st.columns(2)
    with left:
        chart_card("Line plot", ch.monthly_lines_by_region, df, "Max_Temperature_C",
                   note="The seasonal cycle of maximum temperature in each region.",
                   read="a line plot shows change along an ordered axis, here the months of the year.")
        chart_card("Scatter plot", ch.scatter_heatwave, df, "Relative_Humidity_pct", "Solar_Radiation_MJm2day",
                   note="Humidity against solar radiation, heatwave days highlighted.",
                   read="a scatter plot shows how two attributes relate. Heatwave days sit in the dry, sunny corner.")
        chart_card("Box plot", ch.box_by_group, df, "Max_Temperature_C", "Region", ch.REGIONS,
                   note="Spread of maximum temperature in each region.",
                   read="the box is the middle 50% of days and the line inside it is the median.")
    with right:
        chart_card("Bar plot", ch.columns, by_year, "Heatwave days",
                   note="Heatwave days in each year.",
                   read="a bar plot compares one quantity across categories.")
        chart_card("Pie plot", ch.pie, by_region,
                   note="Share of all heatwave days by region.",
                   read="a pie plot shows parts of a whole. It only works with a few slices, so small regions are grouped.")
        chart_card("Histogram", ch.histogram, df["Max_Temperature_C"], an.LABELS["Max_Temperature_C"],
                   markers=threshold, note="Distribution of the daily maximum temperature.",
                   read="a histogram counts how many values fall in each range. Only days right of the line are heatwaves.")


# ================================================================ 11. EXPERIMENT 8

elif page == "Visualization techniques":
    hero("Experiment 8", "Visualization techniques",
         "One example from each of the four families: pixel-oriented, geometric projection, "
         "icon-based and hierarchical.")

    section("1 · Pixel-oriented")
    chart_card("Every record as one pixel", ch.pixel_map, df, "Max_Temperature_C",
               note="Rows are cities grouped by region, columns are days, colour is the maximum temperature.",
               read="the vertical stripes are the seasons repeating every year.")
    chart_card("Heat map", ch.month_region_heatmap, df,
               note="Number of heatwave days for every region and month.")

    section("2 · Geometric projection")
    left, middle, right = st.columns([1, 4, 1])
    with middle:
        chart_card("3D scatter plot", ch.scatter_3d, df, "Max_Temperature_C", "Relative_Humidity_pct", "Solar_Radiation_MJm2day",
                   note="Three attributes projected onto the flat screen.",
                   read="heatwave days form a tight group: hot, dry and sunny at the same time.")

    section("3 · Icon-based")
    chart_card("Glyph map", ch.glyph_map, df,
               note="One circle per city at its latitude and longitude. Size = heatwave days, colour = mean max temperature.")
    chart_card("Icon array", ch.icon_array, df,
               note="Each region drawn as 100 icons. One icon is 1% of its days.")

    section("4 · Hierarchical")
    chart_card("Tree map", ch.treemap, df,
               note="Region → City. The area of each rectangle is its number of heatwave days.")
    chart_card("Dendrogram", ch.city_dendrogram, df, an.NUMERIC_COLS, maths=an.z_score_normalize,
               note="Hierarchical clustering of the cities by their z-score normalized average weather.",
               read="cities that join early, towards the right, have similar climates.")
