# Heatwave Intelligence Dashboard — Complete Project Guide

This guide explains the whole project from the ground up: the Python you need to read the code,
what each section of the dashboard does, the maths behind it, and every Pandas, NumPy, Matplotlib
and Streamlit command that is used. Formulas are written in LaTeX; open this file in the VS Code
Markdown preview (`Cmd+Shift+V`) or on GitHub to see them rendered.

**How to use it.** Read sections 1 to 3 first; they are short and everything else builds on them.
Then read one experiment section (5 to 12) at a time with that page of the dashboard open beside
you. Sections 13 and 14 explain the code file by file. Sections 17 and 18 are for the presentation.

**Contents**

*Part A — Foundations*

1. [The project in one page](#1-the-project-in-one-page)
2. [The Python you need to read the code](#2-the-python-you-need-to-read-the-code)
3. [How the data flows through the project](#3-how-the-data-flows-through-the-project)
4. [The dataset](#4-the-dataset)

*Part B — The eight experiments*

5. [Experiment 1 — Attribute types](#5-experiment-1--attribute-types)
6. [Experiment 2 — Pandas and NumPy data handling](#6-experiment-2--pandas-and-numpy-data-handling)
7. [Experiment 3 — Central tendency and variability](#7-experiment-3--central-tendency-and-variability)
8. [Experiment 4 — Correlation coefficient](#8-experiment-4--correlation-coefficient)
9. [Experiment 5 — Predicting missing values with regression](#9-experiment-5--predicting-missing-values-with-regression)
10. [Experiment 6 — Normalization and K-means discretization](#10-experiment-6--normalization-and-k-means-discretization)
11. [Experiment 7 — The six Matplotlib plots](#11-experiment-7--the-six-matplotlib-plots)
12. [Experiment 8 — Visualization techniques](#12-experiment-8--visualization-techniques)

*Part C — The code*

13. [How the dashboard itself works (Streamlit)](#13-how-the-dashboard-itself-works-streamlit)
14. [File-by-file walkthrough](#14-file-by-file-walkthrough)
15. [Python, Weka and RapidMiner](#15-python-weka-and-rapidminer)

*Part D — Revision and presenting*

16. [Command cheat sheets](#16-command-cheat-sheets)
17. [Presenting it: a five-minute demo](#17-presenting-it-a-five-minute-demo)
18. [Likely viva questions](#18-likely-viva-questions)
19. [Glossary](#19-glossary)

---

## 1. The project in one page

**Goal.** Take a weather dataset and use the FDS lab techniques to answer: *where, when and under
what conditions do heatwaves happen?*

**What it is.** A web dashboard with nine pages: an Overview, one page for each lab experiment
(Experiments 1 and 2 share a page), and a Regression simulator for trying Experiment 5 on any
attribute. It runs in the browser but is written entirely in
Python.

**The files.**

| File | Role | Libraries |
|---|---|---|
| `analysis.py` | Every statistic, written out from its formula | NumPy, Pandas, `math` |
| `charts.py` | Every Matplotlib figure | Matplotlib (plus SciPy for the dendrogram) |
| `app.py` | The web page: puts the charts and tables on screen, top to bottom | Streamlit |
| `style.css` | How the page looks: colours, panels, spacing, the flip animation | CSS |
| `fonts/` | The Josefin Sans font file the charts use | – |
| `files/heatwave_dataset_us.csv` | The dataset | – |
| `.streamlit/config.toml` | Streamlit's theme colours and font | – |
| `requirements.txt` | The libraries to install, with their versions | – |

**Why the statistics are hand-written.** The lab write-ups say "without using ready-made
functions". So `analysis.py` never calls `np.mean`, `np.std`, `np.corrcoef` or scikit-learn. It
only uses basic array arithmetic (`+ - * /`, `.sum()`, `np.sort`) and builds each formula from
that. Two honest exceptions: the multiple regression uses `np.linalg.inv` for the matrix inverse,
and the dendrogram uses SciPy's `linkage`.

**The main findings.**

| Finding | Number |
|---|---|
| Heatwaves are rare | 2,574 of 102,256 city-days (2.5%) |
| One region dominates | Southwest Desert has 60% of all heatwave days |
| A few cities dominate | Yuma has 393; 25 of 56 cities have none |
| They are seasonal | 88% fall in June–August; July is the peak |
| They are increasing | 363 days in 2019 → 701 in 2023 (+93%) |
| Heatwave days are dry and sunny | humidity 28% vs 64%; solar radiation 27.1 vs 17.0 MJ/m²/day |
| Temperature follows sunshine | r = +0.66 with solar radiation, r = −0.44 with humidity |

**Run it.**

```bash
source .venv/bin/activate
streamlit run app.py        # the dashboard, at http://localhost:8501
python analysis.py          # the same results printed in the terminal
```

---

## 2. The Python you need to read the code

Everything in the project is built from the handful of ideas below. If you can read these, you
can read all three Python files.

### Variables, numbers and text

```python
k = 4                              # a whole number (int)
threshold = 40.0                   # a decimal number (float)
region = "Southwest Desert"        # text (str)
is_hot = threshold >= 40           # True or False (bool)
```

**f-strings** put values inside text. The part after `:` is the format:

```python
f"{len(df):,} records"             # 102,256 records     (, = thousands separator)
f"{21.6661:.2f} °C"                # 21.67 °C            (.2f = two decimals)
f"{0.6638:+.2f}"                   # +0.66               (+ = always show the sign)
f"{7:02d}"                         # 07                  (pad to two digits)
```

### Lists, tuples and dictionaries

```python
months = ["Jan", "Feb", "Mar"]     # list: ordered, uses [ ]
months[0]                          # "Jan"   (counting starts at 0)
months[-1]                         # "Mar"   (negative = from the end)
months[0:2]                        # ["Jan", "Feb"]   (a slice: start included, stop excluded)
values[::-1]                       # the whole list reversed
values[::30]                       # every 30th item

point = (33.45, -112.07)           # tuple: like a list but cannot be changed
lat, lon = point                   # "unpacking": two names in one line

labels = {"Wind_Speed_mps": "Wind speed (m/s)"}     # dictionary: key -> value
labels["Wind_Speed_mps"]           # "Wind speed (m/s)"
labels.get("unknown", "n/a")       # "n/a"   (.get gives a default instead of an error)
```

`analysis.py` uses dictionaries for the readable column names (`LABELS`, `SHORT`) and to return
several results at once (`imputation_experiment` returns a dictionary).

### Conditions and loops

```python
if n % 2 == 0:                     # % is the remainder, so this means "n is even"
    middle = (x[n // 2 - 1] + x[n // 2]) / 2     # // is division rounded down
else:
    middle = x[n // 2]

for c in range(k):                 # c = 0, 1, ..., k-1
    ...
for i, name in enumerate(names):   # i = position, name = item
    ...
for label, value in table.items(): # walk through a dictionary or a Pandas Series
    ...
```

A **list comprehension** builds a list in one line:

```python
[f"Bin {c + 1}" for c in labels]                 # ["Bin 1", "Bin 3", ...]
[r for r in REGIONS if r in present]             # keep only some items
```

### Functions

```python
def mean(values):                  # def name(inputs):
    x = np.asarray(values, dtype=float)
    return x.sum() / len(x)        # return hands the answer back
```

- **Default values:** `def kmeans(points, k, max_iter=100, seed=42)` means `max_iter` and `seed`
  can be left out when calling it.
- **Keyword arguments:** `ax.hist(values, bins=40, color=TEAL)` names the inputs, so the order
  does not matter.
- **Several results:** `return labels, centroids, wcss, iteration` returns four things, received
  with `labels, centroids, wcss, iterations = kmeans(...)`.
- **`*args` and `**kwargs`:** "any extra inputs". `chart_card(title, plot, *args, **kwargs)`
  accepts whatever follows `plot` and passes it straight on with `plot(*args, **kwargs)`.
- **A function can be passed like a value.** `chart_card("...", ch.columns, by_year, "Heatwave days")`
  hands over the function `ch.columns` itself (no brackets), and `chart_card` calls it later.
- **`lambda`** is a one-line function with no name: `lambda: export.to_csv(index=False)`.

### Imports

```python
import numpy as np                 # use the library under a short name: np.sort(...)
import analysis as an              # our own file analysis.py: an.mean(...)
import charts as ch                # our own file charts.py:   ch.histogram(...)
from pathlib import Path           # take one thing out of a library
```

### `with` blocks

`with` means "do the indented lines inside this thing":

```python
with st.sidebar:                   # everything indented goes in the sidebar
    st.caption("...")
with card("Frequency table"):      # everything indented goes inside this panel
    st.dataframe(table)
with pd.ExcelWriter(buffer) as writer:   # open the Excel file, write, close it automatically
    df.to_excel(writer)
```

### Decorators

A line starting with `@` above a function changes how the function behaves:

```python
@st.cache_data                     # remember the result; do not recompute next time
def load_data():
    return an.load_data()

@st.fragment                       # when a widget inside changes, re-run only this function
def chart_card(...):
    ...
```

### The Pandas ideas that matter most

- A **DataFrame** (`df`) is the table. A **Series** is one column, or any labelled list of values
  such as the result of a `groupby`.
- `df["Max_Temperature_C"] > 40` gives a column of True/False (a **mask**). `df[mask]` keeps the
  True rows.
- `.to_numpy()` turns a column into a plain NumPy array for arithmetic.
- **Method chaining** reads left to right:
  `heatwaves.groupby("Location_Name").size().sort_values(ascending=False).head(10)` means
  "group by city, count rows, sort largest first, keep the top ten".

---

## 3. How the data flows through the project

```text
files/heatwave_dataset_us.csv
        │   pd.read_csv                                   (analysis.load_data)
        ▼
   data   ── the full table, 102,256 rows
        │   sidebar filters: Year / Region / Season       (app.py, block 3)
        ▼
    df    ── the rows currently selected
        │
        ├──► analysis.py   numbers and small tables   (mean, r, regression, K-means, ...)
        │         │
        │         ▼
        ├──► charts.py     Matplotlib figures         (each function returns one figure)
        │         │
        ▼         ▼
      app.py   puts numbers, tables and figures on the page   (st.dataframe, st.pyplot, ...)
        │
        ▼
   the browser, styled by style.css
```

**What happens when you click something.**

1. You change a widget, for example pick "2023" in the Year filter.
2. Streamlit re-runs `app.py` from the first line to the last.
3. `load_data()` is cached, so the CSV is not read again.
4. `df` is rebuilt with only the 2023 rows.
5. The block for the current page runs: it calls functions in `analysis.py` for the numbers and
   functions in `charts.py` for the figures, using the new `df`.
6. The page in the browser updates.

So a filter is not connected to each chart one by one. It changes `df`, and every chart is drawn
from `df`.

**One chart, start to finish** — "Heatwave days by region" style bar chart:

```python
heatwaves = df[df["Is_Heatwave"] == 1]                         # 1. keep the heatwave days
by_region = heatwaves.groupby("Region").size()                 # 2. count them per region
by_region = by_region.sort_values(ascending=False)             # 3. largest first
chart_card("Heatwave days by region", ch.hbar, by_region, "Heatwave days")   # 4. draw
```

Inside step 4, `chart_card` calls `ch.hbar(by_region, "Heatwave days")`. That function creates a
Matplotlib figure, draws one bar per region and returns the figure. `chart_card` then shows it
with `st.pyplot(figure)`.

---

## 4. The dataset

`files/heatwave_dataset_us.csv` — one row per city per day.

- **102,256 rows × 19 columns**, no missing values, no duplicate rows.
- **56 cities** in **7 regions** (8 cities each), every day from 1 Jan 2019 to 31 Dec 2023
  (1,826 days × 56 cities = 102,256).
- Two label columns are derived from the maximum temperature:

| `Heat_Severity_Class` | Rule | Records | `Heatwave_Flag` |
|---|---|---|---|
| Normal | max temp < 35 °C | 92,638 | No |
| Warm | 35 – 40 °C | 7,044 | No |
| Hot | 40 – 45 °C | 2,391 | Yes |
| Severe | ≥ 45 °C | 183 | Yes |

So **a heatwave day is simply a day whose maximum temperature reaches 40 °C**. The code adds one
helper column, `Is_Heatwave` (1 for Yes, 0 for No), so heatwave days can be counted with a sum.

```python
df = pd.read_csv(path)                                    # load the CSV into a DataFrame
df["Date"] = pd.to_datetime(df["Date"])                   # text -> real dates
df["Is_Heatwave"] = (df["Heatwave_Flag"] == "Yes").astype(int)   # True/False -> 1/0
```

---

## 5. Experiment 1 — Attribute types

**Idea.** Before analysing a column you must know what kind of values it holds, because that
decides which operations make sense.

**Two ways to classify an attribute.**

| Scale | What you can do | Example here |
|---|---|---|
| **Nominal** | only check equal / not equal | `Region`, `Season`, `Location_Name`, `Heatwave_Flag` |
| **Ordinal** | also put in order (but gaps are not measurable) | `Heat_Severity_Class` (Normal < Warm < Hot < Severe), `Month_Name` |
| **Interval** | also add and subtract; **zero is arbitrary** | temperatures in °C, `Latitude`, `Longitude`, `Year`, `Date` |
| **Ratio** | also multiply and divide; **zero means "none"** | `Wind_Speed_mps`, `Precipitation_mm`, `Relative_Humidity_pct`, pressure, solar radiation |

| Type | Meaning | Example here |
|---|---|---|
| **Discrete** | countable set of values | `Year`, `Month`, all the categorical columns |
| **Continuous** | any real number in a range | temperatures, humidity, wind speed |

**The test for interval vs ratio.** Ask "does 0 mean *nothing*, and does *twice as much* make
sense?" 20 °C is not twice as hot as 10 °C (0 °C is just where water freezes), so temperature in
°C is interval. 4 m/s wind is twice 2 m/s, and 0 m/s means no wind, so wind speed is ratio.

**Why it matters for the rest of the project.**

- Mean and standard deviation need at least an interval scale. For a nominal column only the mode
  is meaningful.
- The coefficient of variation (std ÷ mean) needs a ratio scale. That is why the CV for
  temperatures in the dashboard should be read with care (see section 7).
- Correlation and regression are only run on the numeric columns.

**In the code.** `ATTRIBUTE_TYPES` in `analysis.py` is a list of tuples, one per column. The page
turns it into a table and counts the scales:

```python
attributes = pd.DataFrame(an.ATTRIBUTE_TYPES, columns=[...])      # list of tuples -> table
attributes["Scale"].value_counts()                                 # how many of each scale
```

Result: 4 nominal, 2 ordinal, 8 interval, 5 ratio.

---

## 6. Experiment 2 — Pandas and NumPy data handling

**Idea.** Pandas holds the table (a **DataFrame**: labelled rows and columns). NumPy holds plain
numeric arrays and does fast arithmetic on whole arrays at once.

### Pandas commands used in the project

| Command | What it does | Where it is used |
|---|---|---|
| `pd.read_csv(path)` | read a CSV into a DataFrame | `load_data` |
| `df.shape` | `(rows, columns)` | Dataset page tiles |
| `df.head(n)` | first `n` rows | "df.head()" card |
| `df.describe()` | count, mean, std, min, quartiles, max per numeric column | "df.describe()" card |
| `df.isnull().sum()` | missing values per column | "Missing values" tile |
| `df["col"]` | pick one column (a **Series**) | everywhere |
| `df[df["col"] > t]` | keep rows where the condition is true (boolean mask) | filtering card, sidebar filters |
| `df["col"].isin(list)` | True where the value is in the list | Year / Region / Season filters |
| `df.loc[row, col]` | pick by label | `df.loc[df["Max_Temperature_C"].idxmax()]` = hottest row |
| `df["col"].value_counts()` | frequency of each distinct value | severity-class counts |
| `df.groupby("col").agg(...)` | split into groups, summarise each | region / year / city summaries |
| `df.groupby(...).size()` | number of rows in each group | heatwave days per region |
| `.sort_values(...)` | sort | top-10 cities |
| `.reindex(order, fill_value=0)` | force a fixed order, fill gaps with 0 | so a region with 0 heatwave days still appears |
| `pd.crosstab(a, b)` | two-way frequency table | K-means bin vs severity class |
| `df.pivot_table(index, columns, values)` | reshape long data into a grid | the pixel map and the heat map |
| `df.sample(n, random_state=42)` | random rows, reproducible | scatter plots (3,000 points instead of 102,256) |
| `df.to_csv()` / `df.to_excel()` | export | download buttons |

**How a filter works.** A condition on a column gives a column of True/False values (a *mask*).
Putting the mask inside `df[...]` keeps only the True rows:

```python
df = data
if years:
    df = df[df["Year"].isin(years)]          # keep only the chosen years
```

**How groupby works** — "split, apply, combine":

```python
df.groupby("Region").agg(
    Records=("Max_Temperature_C", "size"),        # rows per region
    Mean_max_temp=("Max_Temperature_C", "mean"),  # average of that column per region
    Heatwave_days=("Is_Heatwave", "sum"),         # 1s add up to a count
)
```

### NumPy commands used in the project

| Command | What it does |
|---|---|
| `np.asarray(x, dtype=float)` | turn a list or Series into a NumPy array |
| `x.sum()`, `x.min()`, `x.max()` | add up / smallest / largest |
| `np.sort(x)` | sorted copy (needed for the median and quartiles) |
| `np.abs(x)` | absolute value of every element |
| `x - m`, `x ** 2`, `x / s` | arithmetic on every element at once (**vectorisation**) |
| `x[mask]` | keep elements where the mask is True |
| `np.column_stack([a, b])` | put 1-D arrays side by side as columns of a matrix |
| `np.ones(n)`, `np.zeros(n)`, `np.full(n, v)` | arrays filled with 1, 0 or `v` |
| `X.T` | transpose of a matrix |
| `A @ B` | matrix multiplication |
| `np.linalg.inv(A)` | inverse of a matrix |
| `np.argsort(x)` | the order that would sort `x` |
| `d.argmin(axis=1)` | position of the smallest value in each row |
| `np.random.default_rng(42)` | a random generator with a fixed seed, so results repeat |
| `rng.choice(n, size=k, replace=False)` | pick `k` different random positions |
| `np.meshgrid`, `np.linspace`, `np.arange`, `np.cumsum` | grids, evenly spaced values, running totals |

**Broadcasting.** `x - m` subtracts the single number `m` from every element of the array `x`
without a loop. NumPy "stretches" the smaller shape to match the bigger one. The K-means code uses
this to get every point's distance to every centroid in one line:

```python
dist = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
# X is (n, 1, d), centroids is (1, k, d)  ->  result is (n, k): n points x k centroids
```

---

## 7. Experiment 3 — Central tendency and variability

**Idea.** Summarise a whole column with a few numbers: where its centre is, and how spread out it
is. All formulas are in `analysis.py`.

### The maths

For values $x_1, x_2, \dots, x_n$:

| Measure | Formula | Meaning |
|---|---|---|
| **Mean** | $\bar{x} = \dfrac{1}{n}\sum_{i=1}^{n} x_i$ | the balance point |
| **Median** | middle value of the sorted data; for even $n$, the average of the two middle values | half the data is below it |
| **Mode** | the value that occurs most often | the most common value |
| **Range** | $x_{\max} - x_{\min}$ | total width |
| **Variance** | $\sigma^2 = \dfrac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2$ | average squared distance from the mean |
| **Standard deviation** | $\sigma = \sqrt{\sigma^2}$ | typical distance from the mean, in the original units |
| **Quartiles** | $Q_1$ = median of the lower half, $Q_3$ = median of the upper half | cut the data into quarters |
| **Interquartile range** | $\text{IQR} = Q_3 - Q_1$ | width of the middle 50% |
| **Coefficient of variation** | $\text{CV} = \dfrac{\sigma}{\bar{x}} \times 100\%$ | spread relative to the mean |

**Worked example** (the sample from your Exp 3 write-up): `8, 12, 12, 15, 19, 21, 24, 24, 24, 30`, $n = 10$.

- Mean $= 189 / 10 = 18.9$
- Median: $n$ is even, so average the 5th and 6th values $= (19 + 21)/2 = 20$
- Mode $= 24$ (appears 3 times)
- Variance $= \dfrac{(8-18.9)^2 + \dots + (30-18.9)^2}{10} = \dfrac{434.9}{10} = 43.49$
- Standard deviation $= \sqrt{43.49} = 6.59$
- Lower half `8, 12, 12, 15, 19` → $Q_1 = 12$; upper half `21, 24, 24, 24, 30` → $Q_3 = 24$; IQR $= 12$
- Range $= 30 - 8 = 22$; CV $= 6.59 / 18.9 = 34.9\%$

**Population vs sample.** The code divides by $n$ (population variance). `df.describe()` divides
by $n-1$ (sample variance). With 102,256 rows the two agree to two decimals, which is why the
dashboard's "our functions vs NumPy" check matches.

### The code

```python
def mean(values):
    x = np.asarray(values, dtype=float)
    return x.sum() / len(x)                      # sum of values / number of values

def median(values):
    x = np.sort(np.asarray(values, dtype=float)) # sort first
    n = len(x)
    if n % 2 == 0:
        return (x[n // 2 - 1] + x[n // 2]) / 2   # even: average the two middle values
    return x[n // 2]                             # odd: the middle value

def variance(values):
    x = np.asarray(values, dtype=float)
    m = mean(x)
    return ((x - m) ** 2).sum() / len(x)         # mean of squared deviations
```

The mode uses a dictionary as a tally: `counts[v] = counts.get(v, 0) + 1`, then picks the value
with the largest count. `//` is integer division, `%` is the remainder.

### Grouped data (class intervals)

When data is given as a frequency table instead of raw values, each class is represented by its
**midpoint** $m_i$ with frequency $f_i$, class width $h$ and $N = \sum f_i$.

| Measure | Formula |
|---|---|
| Mean | $\bar{x} = \dfrac{\sum f_i m_i}{N}$ |
| Variance | $\sigma^2 = \dfrac{\sum f_i (m_i - \bar{x})^2}{N}$ |
| Median | $L + \dfrac{N/2 - cf}{f} \times h$ |
| Mode | $L + \dfrac{f_1 - f_0}{2f_1 - f_0 - f_2} \times h$ |

- Median: $L$ = lower boundary of the class containing the $N/2$-th value, $cf$ = cumulative
  frequency *before* that class, $f$ = its frequency.
- Mode: $L$ = lower boundary of the **modal class** (highest frequency $f_1$); $f_0$ and $f_2$ are
  the frequencies of the classes before and after it.
- $Q_1$ and $Q_3$ use the median formula with $N/4$ and $3N/4$.

**On the real data** (max temperature, 8 classes of width 10):

| Class | −27–−17 | −17–−7 | −7–3 | 3–13 | 13–23 | **23–33** | 33–43 | 43–53 |
|---|---|---|---|---|---|---|---|---|
| Frequency | 53 | 425 | 5,688 | 16,510 | 28,626 | **36,070** | 14,182 | 702 |

- Median class is 13–23 ($cf = 22{,}676$ before it):
  $13 + \dfrac{51{,}128 - 22{,}676}{28{,}626}\times 10 = 22.94$
- Modal class is 23–33:
  $23 + \dfrac{36{,}070 - 28{,}626}{2(36{,}070) - 28{,}626 - 14{,}182}\times 10 = 25.54$

| Measure | From grouped data | From raw data |
|---|---|---|
| Mean | 21.63 | 21.67 |
| Median | 22.94 | 22.94 |
| Mode | 25.54 | 29.32 |
| Std dev | 11.26 | 10.89 |

Grouping loses detail (every value is replaced by its class midpoint), so the answers are close
but not identical. The mode differs most because the raw mode is a single repeated reading while
the grouped mode is the peak of the histogram.

### What the results say

| Attribute | Mean | Median | Std dev | Reading |
|---|---|---|---|---|
| Max temperature (°C) | 21.67 | 22.94 | 10.89 | mean < median → longer tail on the cold side (negatively skewed) |
| Relative humidity (%) | 63.49 | 68.12 | 20.78 | also negatively skewed |
| Precipitation (mm) | 2.06 | 0.04 | 5.68 | mean ≫ median → most days are dry, a few very wet days pull the mean up; CV = 276% |
| Surface pressure (kPa) | 95.56 | 98.26 | 6.67 | very stable, CV = 7% |

**Rule of thumb:** mean ≈ median → symmetric; mean < median → negative skew; mean > median →
positive skew. When the data is skewed, the median is the better "typical" value.

**Caution on CV.** CV divides by the mean, so it only makes sense on a ratio scale. For
temperatures in °C the mean could be near 0 and the CV would explode, so compare CV only between
ratio attributes (wind, precipitation, humidity, pressure).

---

## 8. Experiment 4 — Correlation coefficient

**Idea.** One number, $r$, that says how strongly two attributes move together **in a straight
line**.

### The maths

Pearson's correlation coefficient:

$$r = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2 \;\sum_{i=1}^{n}(y_i-\bar{y})^2}}$$

- **Numerator.** For each row, multiply "how far $x$ is from its mean" by "how far $y$ is from its
  mean". If both are above (or both below) their means the product is positive; if one is above
  and the other below it is negative. Adding these up tells you which pattern wins.
- **Denominator.** Rescales the result so it always lies between −1 and +1, whatever the units.
- Equivalent form: $r = \dfrac{\text{cov}(x,y)}{\sigma_x \sigma_y}$.

| Value of $r$ | Meaning | Word used in the dashboard |
|---|---|---|
| +1 | perfect straight line going up | |
| $\lvert r\rvert \ge 0.7$ | strong | strong positive / negative |
| $0.4 \le \lvert r\rvert < 0.7$ | moderate | moderate positive / negative |
| $0.2 \le \lvert r\rvert < 0.4$ | weak | weak positive / negative |
| $\lvert r\rvert < 0.2$ | none | no linear relationship |
| −1 | perfect straight line going down | |

### The code

```python
def correlation_coefficient(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x - mean(x)                              # deviations of x
    dy = y - mean(y)                              # deviations of y
    denominator = math.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    return (dx * dy).sum() / denominator
```

`correlation_matrix` runs this for every pair of columns and stores the results in a square
table. The matrix is symmetric ($r_{xy} = r_{yx}$) with 1s on the diagonal ($r_{xx} = 1$), so the
loop only computes the upper triangle and copies it.

### What the results say

| Pair | $r$ | Reading |
|---|---|---|
| Max temp vs Mean temp | +0.98 | nearly the same information |
| Max temp vs Min temp | +0.92 | hot days have warm nights |
| Max temp vs Solar radiation | **+0.66** | sunnier days are hotter (the "positive" example) |
| Max temp vs Humidity | **−0.44** | hotter days are drier |
| Latitude vs Min temp | **−0.44** | further north, colder nights (the "negative" example) |
| Wind speed vs Precipitation | **+0.07** | unrelated (the "none" example) |
| Solar radiation vs Humidity | −0.53 | cloudy, humid days get less sun |

### Limits to remember

1. **Only straight-line relationships.** A perfect U-shape can have $r = 0$.
2. **Correlation is not causation.** Humidity and temperature move together, but that alone does
   not prove one causes the other.
3. **Outliers** can change $r$ a lot.

---

## 9. Experiment 5 — Predicting missing values with regression

**Idea.** If an attribute is missing in some rows but is correlated with other attributes, fit a
line through the rows where it *is* known, and use the line to predict the gaps.

### How the experiment is set up

The dataset has no missing values, so the code creates them on purpose (this is the only way to
*measure* accuracy, because the true answers are still known):

1. Pick a target attribute $y$ (default: max temperature).
2. Randomly hide 10% of its values (10,225 rows) using a fixed random seed.
3. **Train** on the 90% that are still known.
4. **Predict** the hidden 10% three ways: mean, simple regression, multiple regression.
5. Compare each prediction with the true hidden value.

```python
rng = np.random.default_rng(seed)                       # fixed seed -> same rows every run
hidden = np.zeros(n, dtype=bool)                        # all False
hidden[rng.choice(n, size=int(n * pct / 100), replace=False)] = True
known, missing = ~hidden, hidden                        # ~ flips True/False
```

### Simple linear regression (one predictor)

Model: $\;y = w_0 + w_1 x$

The **method of least squares** picks the line that makes the sum of squared errors as small as
possible:

$$S(w_0, w_1) = \sum_{i=1}^{n}\bigl(y_i - w_0 - w_1 x_i\bigr)^2$$

Setting $\partial S/\partial w_0 = 0$ and $\partial S/\partial w_1 = 0$ and solving gives:

$$w_1 = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sum (x_i-\bar{x})^2} \qquad\qquad w_0 = \bar{y} - w_1\bar{x}$$

- $w_1$ (**slope**): how much $y$ changes when $x$ goes up by 1.
- $w_0$ (**intercept**): the value of $y$ when $x = 0$.
- The line always passes through the point $(\bar{x}, \bar{y})$.
- Link to correlation: $w_1 = r\,\dfrac{\sigma_y}{\sigma_x}$, and for simple regression $R^2 = r^2$.

```python
def simple_linear_regression(x, y):
    x_bar, y_bar = mean(x), mean(y)
    w1 = ((x - x_bar) * (y - y_bar)).sum() / ((x - x_bar) ** 2).sum()
    w0 = y_bar - w1 * x_bar
    return w0, w1
```

**Result** (max temperature from solar radiation):
$\;\text{Max temp} = 5.63 + 0.931 \times \text{Solar radiation}$.
Each extra MJ/m²/day of sunshine goes with about 0.93 °C more.

### Multiple linear regression (several predictors)

Model: $\;y_i = b_0 + b_1 x_{i1} + b_2 x_{i2} + \dots + b_p x_{ip} + \varepsilon_i$

Write it with matrices. $X$ has one row per record and one column per predictor, **plus a first
column of 1s** so that $b_0$ is handled like any other coefficient:

$$Y = Xb \qquad X = \begin{bmatrix} 1 & x_{11} & \cdots & x_{1p} \\ 1 & x_{21} & \cdots & x_{2p} \\ \vdots & & & \vdots \\ 1 & x_{n1} & \cdots & x_{np} \end{bmatrix}$$

$X$ is not square, so it cannot be inverted directly. Multiply both sides by $X^T$ to get the
**normal equations**, whose matrix $X^TX$ *is* square:

$$X^T X\, b = X^T Y \quad\Longrightarrow\quad b = (X^T X)^{-1} X^T Y$$

```python
def multiple_linear_regression(X, y):
    X = np.column_stack([np.ones(len(X)), X])     # add the column of 1s for b0
    return np.linalg.inv(X.T @ X) @ (X.T @ y)     # b = (XᵀX)⁻¹ XᵀY

def predict_multiple(b, X):
    return b[0] + X @ b[1:]                       # intercept + X times the slopes
```

**Result** (4 predictors):

$$\text{Max temp} = 41.84 + 0.760\,\text{Solar} - 0.063\,\text{Humidity} - 0.752\,\text{Latitude} - 0.788\,\text{Wind}$$

Each coefficient is the effect of that predictor **with the others held fixed**. For example, one
degree further north goes with 0.75 °C lower maximum temperature.

### Measuring accuracy

With actual values $y_i$ and predictions $\hat{y}_i$ over the $m$ hidden rows:

| Metric | Formula | Meaning |
|---|---|---|
| **MAE** | $\dfrac{1}{m}\sum \lvert y_i - \hat{y}_i\rvert$ | average size of the error |
| **RMSE** | $\sqrt{\dfrac{1}{m}\sum (y_i - \hat{y}_i)^2}$ | like MAE but punishes big errors more |
| **R²** | $1 - \dfrac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | share of the variation the model explains (1 = perfect, 0 = no better than the mean) |

### Results and comparison

| Method | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|
| Mean imputation (fill every gap with the mean) | 8.90 | 10.84 | 0.000 |
| Simple linear regression (solar radiation) | 6.55 | 8.13 | 0.437 |
| Multiple linear regression (4 predictors) | 5.93 | 7.34 | 0.542 |

- Mean imputation has $R^2 = 0$ by definition: it predicts the same number for every row.
- Simple regression checks out against the correlation: $r = 0.66 \Rightarrow r^2 \approx 0.44$.
- Multiple regression is best: its RMSE is 32% lower than mean imputation.

| | Simple | Multiple |
|---|---|---|
| Predictors | one | two or more |
| Shape fitted | a line | a plane / hyper-plane |
| Solved by | two closed-form formulas | matrix algebra (normal equations) |
| Accuracy | lower | higher, if the extra predictors carry new information |

**Why the default avoids temperature predictors.** Min and mean temperature correlate at 0.92 and
0.98 with max temperature, so predicting one temperature from another is almost trivial
($R^2 \approx 0.84$ from min temperature alone). The dashboard uses non-temperature
predictors so the two methods can actually be compared.

### Choosing the predictors: the Regression simulator page

The **Regression simulator** page lets you pick any attribute as the target and answers "which
attributes should I use to predict it?". The rule is in `recommend_predictors` in `analysis.py`:

1. Compute r between every other attribute and the target, and sort from strongest to weakest.
2. **Rule 1 — it must be related to the target:** skip the attribute if $\lvert r\rvert < 0.2$.
3. **Rule 2 — it must not repeat a predictor already chosen:** skip it if its correlation with
   any chosen predictor is $\lvert r\rvert \ge 0.85$.

Rule 2 avoids **multicollinearity**: two predictors that carry almost the same information. The
model cannot tell which of them deserves the credit, so the coefficients become unstable and hard
to interpret, and the second one adds almost no accuracy.

**Example: solar radiation is missing.**

| Attribute | r with solar radiation | Decision |
|---|---|---|
| Max temperature | +0.66 | Use |
| Mean temperature | +0.61 | Skip: repeats max temperature (r = 0.98 between them) |
| Relative humidity | −0.53 | Use |
| Min temperature | +0.51 | Skip: repeats max temperature (r = 0.92) |
| Precipitation | −0.29 | Use |
| Longitude, latitude, pressure, wind | −0.20 to −0.08 | Skip: too weak |

With the three chosen predictors (10% of values hidden):

$$\text{Solar radiation} = 14.26 + 0.399\,\text{Max temp} - 0.081\,\text{Humidity} - 0.259\,\text{Precipitation}$$

| Method | RMSE (MJ/m²/day) | R² |
|---|---|---|
| Mean imputation | 7.74 | 0.000 |
| Simple regression (max temperature only) | 5.81 | 0.437 |
| Multiple regression (3 predictors) | 5.25 | 0.540 |

So the model explains about 54% of the variation: a moderate fit, clearly better than the mean.
A sunny day is hot, dry and rainless, which is exactly what the signs of the coefficients say.

**To fill in one missing value**, put that record's known values into the equation. A day with
max temperature 21.67 °C, humidity 63.49% and precipitation 2.06 mm (the averages) gives
$14.26 + 0.399(21.67) - 0.081(63.49) - 0.259(2.06) \approx 17.2$ MJ/m²/day. Step 3 of the page does
this calculation live.

**When no attribute passes the rules** (wind speed is an example: every $\lvert r\rvert$ is below
0.2), regression has nothing to work with and would not beat filling in the mean.

**A limit of the rule.** It only sees straight-line relationships. Solar radiation depends
strongly on the time of year, but it rises to June and falls again, so the month number has
almost no linear correlation with it and the rule ignores it.

**When is regression imputation appropriate?** When the attribute with gaps is numeric, the
relationship with the predictors is roughly linear (check the scatter plot), the correlation is
reasonably strong, and the predictors themselves are not missing in those rows.

---

## 10. Experiment 6 — Normalization and K-means discretization

### Part A — Normalization

**Idea.** Attributes are measured in different units, so their numbers have very different sizes
(pressure ≈ 95, wind ≈ 2). Any method based on distances would be dominated by the big-number
attribute. Normalization rescales every attribute to a common range.

| Method | Formula | Result |
|---|---|---|
| **Min-max** | $v' = \dfrac{v - \min}{\max - \min}\,(new_{max} - new_{min}) + new_{min}$ | exactly in $[0, 1]$ |
| **Z-score** | $v' = \dfrac{v - \bar{x}}{\sigma}$ | mean 0, standard deviation 1 |
| **Decimal scaling** | $v' = \dfrac{v}{10^{\,j}}$, with $j$ the smallest integer such that $\max\lvert v'\rvert < 1$ | between −1 and 1 |

**Worked example** — max temperature of 9.57 °C (Phoenix, 1 Jan 2019). For the column:
min = −26.23, max = 48.10, mean = 21.67, std = 10.89.

- Min-max: $\dfrac{9.57 - (-26.23)}{48.10 - (-26.23)} = \dfrac{35.80}{74.33} = 0.4816$
- Z-score: $\dfrac{9.57 - 21.67}{10.89} = -1.11$ (1.11 standard deviations below average)
- Decimal scaling: the largest absolute value is 48.10; $48.10/10 = 4.81$ (not below 1),
  $48.10/100 = 0.481$ (below 1), so $j = 2$ and $v' = 9.57/100 = 0.0957$

```python
def min_max_normalize(values, new_min=0.0, new_max=1.0):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) * (new_max - new_min) + new_min

def z_score_normalize(values):
    return (x - mean(x)) / std_dev(x)

def decimal_scaling(values):
    max_abs = np.abs(x).max()
    j = 0
    while max_abs / (10 ** j) >= 1:      # keep dividing by 10 until the largest value is below 1
        j += 1
    return x / (10 ** j), j
```

| | Min-max | Z-score | Decimal scaling |
|---|---|---|---|
| Needs to remember | min, max | mean, std | $j$ |
| Outliers | very sensitive (one extreme value squashes the rest) | less sensitive | sensitive |
| New data outside the old range | goes out of $[0,1]$ ("out-of-bounds") | fine | may need a new $j$ |

**All three are linear transformations**, so they move and stretch the axis but do not change the
*shape* of the distribution. The "Same shape, different scale" card shows this: four identical
histograms with different numbers on the axis.

### Part B — K-means discretization

**Idea.** Discretization replaces a continuous value with the label of a bin. Instead of choosing
the bin edges by hand, K-means lets the data choose them: it finds $k$ groups of similar values.

**What K-means minimises** — the within-cluster sum of squares:

$$\text{WCSS} = \sum_{j=1}^{k}\;\sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2$$

where $\mu_j$ is the centroid (mean) of cluster $C_j$ and the distance is Euclidean:
$d(x, \mu) = \sqrt{\sum_d (x_d - \mu_d)^2}$.

**The algorithm.**

1. Choose $k$.
2. Pick $k$ random data points as the starting centroids.
3. **Assign** every point to its nearest centroid.
4. **Update** every centroid to the mean of the points assigned to it.
5. Repeat 3–4 until no point changes cluster (or a maximum number of iterations is reached).

Both steps can only lower the WCSS or leave it unchanged, so the algorithm always stops.

```python
centroids = X[rng.choice(len(X), size=k, replace=False)].copy()       # step 2
for iteration in range(1, max_iter + 1):
    dist = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2) # squared distances, n x k
    new_labels = dist.argmin(axis=1)                                  # step 3: nearest centroid
    if (new_labels == labels).all():                                  # step 5: nothing changed
        break
    labels = new_labels
    for c in range(k):                                                # step 4: move centroids
        members = X[labels == c]
        if len(members):
            centroids[c] = members.sum(axis=0) / len(members)
```

At the end the clusters are renumbered from lowest to highest centroid so "Bin 1" is always the
coldest.

**Choosing $k$: the elbow method.** Run K-means for $k = 1, 2, \dots, 8$ and plot WCSS against
$k$. WCSS always falls as $k$ grows; the "elbow" is where the curve stops falling steeply, meaning
extra clusters no longer help much.

**Result** (max temperature, $k = 4$, converged in 38 iterations, WCSS ≈ 1,179,342):

| Bin | Range (°C) | Centroid (°C) | Records |
|---|---|---|---|
| Bin 1 | −26.23 to 9.45 | 3.46 | 15,556 |
| Bin 2 | 9.46 to 20.47 | 15.46 | 27,755 |
| Bin 3 | 20.48 to 30.07 | 25.50 | 33,849 |
| Bin 4 | 30.08 to 48.10 | 34.65 | 25,096 |

**K-means bins vs the given severity classes.** The dataset's `Heat_Severity_Class` uses fixed
thresholds (35 / 40 / 45 °C) that only split the hot tail: 91% of records are "Normal". K-means
has no thresholds; it splits the *whole* range into natural groups, so all of Warm, Hot and Severe
land together in Bin 4. Neither is wrong: the classes answer "is this dangerous?", K-means answers
"what groups exist in the data?".

**Which attributes suit K-means?** Numeric attributes (a mean and a distance must make sense),
without extreme outliers (they drag centroids), and normalized first if you cluster on several
attributes at once. Here only one attribute is clustered at a time, so normalization is not
needed for the binning itself.

---

## 11. Experiment 7 — The six Matplotlib plots

**How every Matplotlib figure is built.** A **Figure** is the whole image; an **Axes** is one plot
inside it. You create both, call plotting methods on the Axes, then label it.

```python
fig, ax = plt.subplots(figsize=(5.2, 3.0), layout="constrained")   # figure + one axes
ax.plot(x, y, color="#d03b3b")                                     # draw
ax.set_xlabel("..."); ax.set_ylabel("...")                         # label the axes
ax.legend()                                                        # show the legend
fig.savefig("plot.png", dpi=180)                                   # save as an image
```

| Plot | Command | Answers the question | In the dashboard |
|---|---|---|---|
| **Line** | `ax.plot(x, y)` | how does it change over an ordered axis? | monthly mean temperature of each region |
| **Bar** | `ax.bar(x, h)` / `ax.barh(y, w)` | how do categories compare? | heatwave days per year |
| **Scatter** | `ax.scatter(x, y)` | how are two attributes related? | humidity vs solar radiation, heatwave days highlighted |
| **Pie** | `ax.pie(values, labels=...)` | what share of the whole is each part? | share of heatwave days per region |
| **Box** | `ax.boxplot(data)` | where is the middle, how wide is the spread, any outliers? | max temperature per region |
| **Histogram** | `ax.hist(values, bins=40)` | how are the values distributed? | daily maximum temperature |

### Reading a box plot

- The **box** runs from $Q_1$ to $Q_3$ (its height is the IQR).
- The **line inside** is the median.
- The **whiskers** reach the furthest points within $1.5 \times \text{IQR}$ of the box.
- Points beyond the whiskers are drawn individually as **outliers**:
  below $Q_1 - 1.5\,\text{IQR}$ or above $Q_3 + 1.5\,\text{IQR}$.

### Bar plot vs histogram

| | Bar plot | Histogram |
|---|---|---|
| X axis | categories | a continuous number, cut into bins |
| Bars | separated | touching |
| Height | a value for that category | how many values fall in that bin |
| Order of bars | can be rearranged | fixed by the number line |

### Other Matplotlib commands used

| Command | What it does |
|---|---|
| `ax.fill_between(x, y, alpha=0.1)` | shade the area under a line |
| `ax.annotate(text, xy, xytext=...)` | add text at a data point (this is how you "add text to a plot"; `ax.text(x, y, s)` is the simpler version) |
| `ax.bar_label(bars, labels=...)` | write the value at the end of each bar |
| `ax.axvline(x)` | a vertical reference line (mean / median / mode, the 40 °C threshold) |
| `ax.set_xticks(positions, labels)` | choose the tick marks |
| `ax.grid(axis="x", visible=False)` | switch gridlines on or off |
| `ax.margins(...)`, `ax.set_ylim(bottom=0)` | padding and axis limits |
| `ax.set_title(...)`, `fig.suptitle(...)` | titles |
| `ax.legend(loc=..., ncols=...)` | legend position and layout |
| `plt.subplots(1, 3)` | several plots side by side |
| `plt.rcParams.update({...})` | global style (fonts, colours, gridlines) applied to every figure |
| `plt.close(fig)` | free the figure's memory |

---

## 12. Experiment 8 — Visualization techniques

The lab names four families of techniques. The dashboard has at least one of each.

### 1. Pixel-oriented

**Idea.** Every data value becomes one coloured pixel, so a very large dataset fits in one image.

- **Pixel map.** Rows are the 56 cities (grouped by region), columns are the 1,826 days, colour is
  the attribute's value. That is all 102,256 records in one picture.
- **Heat map.** A small grid (7 regions × 12 months) where colour = number of heatwave days.

```python
grid = df.pivot_table(index=["Region", "Location_Name"], columns="Date", values=column)
ax.imshow(grid.to_numpy(), aspect="auto", cmap=HEAT, interpolation="nearest")
fig.colorbar(im, ax=ax)                      # the scale that explains the colours
```

`pivot_table` reshapes the long table into a matrix; `imshow` draws a matrix as an image. The
colour comes from a **colormap**, a function from a number in $[0, 1]$ to a colour. Magnitude uses
a single-hue ramp (light → dark); the correlation matrix uses a **diverging** map (teal ↔ grey ↔
coral) because its values have a meaningful middle at 0.

### 2. Geometric projection

**Idea.** Data with more than two dimensions is projected onto the flat screen.

The 3D scatter plot shows three attributes at once. The camera position is set by two angles,
azimuth and elevation.

```python
ax = fig.add_subplot(projection="3d")        # a 3D axes
ax.scatter(x, y, z, s=5)                     # points in 3D
ax.view_init(elev=22, azim=-58)              # where the camera sits
```

**The maths.** Rotate each point by the azimuth $\theta$ about the vertical axis and by the
elevation $\varphi$ about the horizontal axis, then drop the depth coordinate:

$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} -\sin\theta & \cos\theta & 0 \\ -\sin\varphi\cos\theta & -\sin\varphi\sin\theta & \cos\varphi \end{bmatrix}\begin{bmatrix} x \\ y \\ z \end{bmatrix}$$

This is the orthographic version; Matplotlib does the work for you and by default also adds a
little perspective (far points drawn slightly smaller). The ordinary 2D scatter plot is the simplest geometric projection:
it shows two of the dimensions and ignores the rest.

### 3. Icon-based

**Idea.** Each record (or group) is drawn as a small symbol whose features encode values.

- **Glyph map.** One circle per city at its (longitude, latitude). **Size** = number of heatwave
  days, **colour** = mean max temperature. Two extra attributes are shown on top of the position.
- **Icon array.** Each region is 100 dots; one dot = 1% of its days, coloured by severity class.

```python
ax.scatter(city["lon"], city["lat"], s=size, c=city["max_temp"], cmap=HEAT)   # s = size, c = colour value
```

**The maths of the icon array.** Percentages rarely add to exactly 100 after rounding, so the code
uses the **largest remainder method**: give each class the whole-number part of its percentage,
then hand the leftover icons to the classes with the largest fractional parts.

### 4. Hierarchical

**Idea.** Show data that has levels (here Region → City).

**Tree map.** Each city is a rectangle whose **area is proportional to its heatwave days**; cities
are nested inside their region's rectangle. The layout (`_split` in `charts.py`) is recursive:

1. Sort the items from largest to smallest.
2. Split them into two groups with roughly equal totals.
3. Cut the rectangle along its **longer side** in the same proportion.
4. Repeat inside each piece until every piece holds one item.

**Dendrogram.** A tree showing which cities have similar climates, built by **agglomerative
hierarchical clustering**:

1. Describe each city by its average of the 8 weather attributes.
2. **Z-score normalize** each attribute, so that pressure (≈ 95) does not outweigh wind (≈ 2). This
   is the practical use of Experiment 6.
3. Start with every city as its own cluster.
4. Repeatedly merge the two closest clusters until one remains.

"Closest" uses **Ward linkage**: merge the pair whose union increases the total within-cluster
variance the least. The cost of merging clusters $A$ and $B$ is

$$\Delta(A, B) = \frac{\lvert A\rvert\,\lvert B\rvert}{\lvert A\rvert + \lvert B\rvert}\;\lVert \mu_A - \mu_B \rVert^2$$

```python
profile = df.groupby("Location_Name")[cols].mean()                       # step 1
scaled = np.column_stack([an.z_score_normalize(profile[c]) for c in cols])   # step 2
tree = dendrogram(linkage(scaled, method="ward"), orientation="left", labels=list(profile.index))
```

**Reading it.** The horizontal position of a join is the distance at which two groups merged.
Cities that join far to the right (small distance) are very similar. The coloured dot beside each
name is the city's region, so you can see where the data-driven clusters agree with the region
labels (most Southwest Desert cities end up in one cluster) and where they do not.

---

## 13. How the dashboard itself works (Streamlit)

**The Streamlit model.** `app.py` is an ordinary Python script. Streamlit runs it from top to
bottom to draw the page, and **re-runs the whole script every time you touch a widget**. Widget
values are remembered between runs in `st.session_state`.

`app.py` is written in the order the page is built, in numbered blocks:

| Block | What it does |
|---|---|
| 1. The look | loads `style.css` and names the three accent colours |
| 2. Small helpers | short functions that draw a banner, tiles, a card, a chart card |
| 3. Sidebar | navigation, the three filters, the download buttons |
| 4 – 11. One block per page | Overview, then one block for each experiment |

### Streamlit commands used

| Command | What it does |
|---|---|
| `st.set_page_config(layout="wide")` | page title, icon and width |
| `st.markdown(html, unsafe_allow_html=True)` | put custom HTML on the page (banner, tiles, headings) |
| `st.write`, `st.caption` | text |
| `st.latex(r"...")` | a rendered formula |
| `st.dataframe(df)` | an interactive, sortable table |
| `st.pyplot(figure)` | show a Matplotlib figure |
| `st.area_chart(table)`, `st.bar_chart(table)` | Streamlit's built-in live charts (hover, zoom) |
| `st.code(source, language="python")` | syntax-highlighted code |
| `st.columns(2)`, `st.container(border=True)` | layout: side-by-side columns, a bordered box |
| `st.sidebar` | the left panel |
| `st.radio`, `st.selectbox`, `st.multiselect`, `st.slider`, `st.toggle` | input widgets |
| `st.download_button(label, data, file_name)` | download a file |
| `@st.cache_data` | remember a function's result so it is not recomputed on every re-run |
| `@st.fragment` | re-run only this function, not the whole page, when a widget inside it changes |
| `st.session_state` | values that survive between re-runs |
| `st.rerun()`, `st.stop()` | restart / stop the current run |

### The helper functions

| Helper | What it draws |
|---|---|
| `hero(eyebrow, title, text, tiles)` | the heading at the top of each page |
| `tiles_html(items)` | a row of coloured, underlined numbers from a list of `(label, value, note)` |
| `section(title)` | a section heading with a line |
| `card(title, note)` | a bordered box for tables and text, used as `with card("Title"):` |
| `chart_card(title, plot, *args)` | a bordered box with one chart and a Code switch |
| `reading(text)` | the "Reading it" explanation under a chart |
| `in_words(number)` | a number written out, e.g. 88 → "Eighty-Eight" |
| `bars_html(series)` | gradient bars built from a Pandas Series |
| `ranking_html(series)` | the city ranking: top three large, the rest as a list |

Every chart on the site is one line. For example:

```python
chart_card("Heatwave days by year", ch.columns, by_year, "Heatwave days")
#           title                    function   inputs to that function
```

`chart_card` calls `ch.columns(by_year, "Heatwave days")`, which returns a Matplotlib figure, and
shows it with `st.pyplot`.

### Filters

The three sidebar `multiselect`s return lists. The DataFrame is filtered once near the top of the
script and every page uses the filtered `df`, so one set of filters controls everything:

```python
df = data
if years:
    df = df[df["Year"].isin(years)]
```

### Live charts on the Overview

Three charts on the Overview are not Matplotlib pictures. They use the charts **built into
Streamlit**, so no extra library is needed, and they respond to the mouse: hover to read a value,
drag to move, scroll to zoom, double-click to reset.

| Chart | Command | Function in `app.py` |
|---|---|---|
| Hot Days Per Month | `st.area_chart(table, stack=True)` | `live_hot_days` |
| Heatwave Days by Year | `st.bar_chart(table, x=..., y=...)` | `live_days_by_year` |
| How Often Each Region Gets Hot | `st.bar_chart(table, horizontal=True, stack=True)` | `live_region_share` |

Each function prepares a small table with Pandas and hands it to Streamlit. For example,
`pd.crosstab(month, severity_class)` counts the days for every month and class, giving one row
per month and one column per class, which is exactly the shape `st.area_chart` expects.

Every other chart on the site is a Matplotlib figure, as Experiment 7 requires.

### The Code switch (card flip)

Each chart card has a `st.toggle("Code")`. When it is on, the card shows the source code of the
chart function instead of the chart:

```python
if show_code:
    st.code(inspect.getsource(plot), language="python")   # the text of the function itself
else:
    figure = plot(*args, **kwargs)
    st.pyplot(figure)
```

- `inspect.getsource(function)` is a standard-library call that returns a function's code as text.
- If a `maths=` function is given (for example `maths=an.kmeans`), its code is shown as well.
- **The flip** is pure CSS. When the switch is on, the card gets a name ending in `--flip`, and
  `style.css` plays a `rotateY` animation on any card with that name.
- `@st.fragment` on `chart_card` means flipping one card redraws only that card.

### The look (`style.css` and one colour theme)

The dashboard has a single theme: deep navy, with **teal, coral and purple** as the accent colours.

- **Page colours** are in `style.css`, written as CSS variables (`var(--panel)`, `var(--teal)`, ...)
  so each colour is defined once and reused.
- **Chart colours** are plain constants at the top of `charts.py` (`TEAL`, `CORAL`, `PURPLE`, ...)
  and are applied to every figure through `plt.rcParams`.
- **The typeface** is Josefin Sans. The page loads it from Google Fonts (set in
  `.streamlit/config.toml`); the charts load the copy in the `fonts/` folder with
  `font_manager.fontManager.addfont(...)`, so page and charts match.
- **Gradient bars.** `_fade` in `charts.py` fills each bar with an image that runs from the
  background colour to the bar's colour (`ax.imshow` with a two-colour colormap). On the page, the
  HTML bars use the CSS `linear-gradient(...)` for the same effect.
- **The Overview's three panels** are built from short helpers in `app.py`: `in_words(88)` gives
  "Eighty-Eight", `bars_html(series)` draws the gradient bars and `ranking_html(series)` draws the
  city ranking. Each one just builds a string of HTML from a Pandas Series.
- Panels and headings fade up when a page opens (`@keyframes rise`).
- There is **no fullscreen button** on charts: `style.css` hides it, because a chart blown up to
  full screen inside an animated panel did not display properly.

### Caching

`@st.cache_data` is used four times: for loading the CSV, the summary table, K-means and the
elbow curve. Streamlit stores the result and returns it instantly when the function is called
again with the same inputs.

### Downloads

The CSV button uses `DataFrame.to_csv(index=False)`. The Excel button uses `pd.ExcelWriter` with
the `xlsxwriter` engine and writes two sheets: the data and the summary statistics. Both export
the **currently filtered** records. The file is only built when you click, because the button is
given a function (`lambda: ...`) rather than ready-made data.

---

## 14. File-by-file walkthrough

### `analysis.py` — the statistics

The file starts with constants (plain data, no logic):

| Name | What it holds |
|---|---|
| `NUMERIC_COLS` | the eight numeric weather columns that the statistics run on |
| `LABELS`, `SHORT` | dictionaries: column name → readable label (long and short) |
| `MONTHS`, `SEASONS`, `SEVERITY_ORDER` | fixed orders, so charts never sort alphabetically |
| `ATTRIBUTE_TYPES` | the Experiment 1 classification table, one tuple per column |

Then the functions, grouped by experiment:

| Function | Takes | Gives back | Explained in |
|---|---|---|---|
| `find_data_file()` | – | the path of the CSV (looks beside the script, then in `files/`) | – |
| `load_data()` | – | the DataFrame, with real dates and the `Is_Heatwave` column | 4 |
| `mean`, `median`, `mode` | a column | one number | 7 |
| `variance`, `std_dev` | a column | one number | 7 |
| `quartiles`, `iqr` | a column | Q1 and Q3; their difference | 7 |
| `value_range`, `coefficient_of_variation` | a column | one number | 7 |
| `summary_table(df)` | the table | one row of statistics per numeric column | 7 |
| `grouped_frequency(values, n_classes)` | a column, number of classes | the frequency table, the grouped statistics, the class edges | 7 |
| `correlation_coefficient(x, y)` | two columns | r | 8 |
| `correlation_matrix(df, cols)` | the table, column names | a square table of r values | 8 |
| `describe_r(r)` | r | words, e.g. "moderate negative" | 8 |
| `simple_linear_regression(x, y)` | predictor, target | `w0, w1` | 9 |
| `multiple_linear_regression(X, y)` | predictor matrix, target | the coefficient array `b` | 9 |
| `predict_multiple(b, X)` | coefficients, predictors | predicted values | 9 |
| `regression_scores(actual, predicted)` | two arrays | MAE, RMSE, R² | 9 |
| `imputation_experiment(...)` | table, target, predictors, % hidden | a dictionary with every result of the experiment | 9 |
| `recommend_predictors(df, target, candidates)` | table, target, candidate columns | a table of r values with a Use / Skip decision, and the list of chosen predictors | 9 |
| `min_max_normalize`, `z_score_normalize` | a column | the rescaled column | 10 |
| `decimal_scaling(values)` | a column | the rescaled column and `j` | 10 |
| `kmeans(points, k)` | values, number of clusters | labels, centroids, WCSS, iterations | 10 |
| `elbow_wcss(points, k_max)` | values | the WCSS for k = 1 … k_max | 10 |
| `cluster_intervals(values, labels, k)` | values and their labels | a table: the range, centroid and size of each bin | 10 |
| `key_insights(df)` | the table | a list of (title, sentence) pairs for the Overview | 1 |

**`key_insights` in plain words.** It computes each headline from `df` and writes a sentence with
an f-string, so the text always matches the current filters. For example the "increasing" insight
counts heatwave days in the first and last year and reports the percentage change:
$\text{change} = \dfrac{\text{last} - \text{first}}{\text{first}} \times 100 = \dfrac{701 - 363}{363} \times 100 = +93\%$.

**The last block**, `if __name__ == "__main__":`, only runs when you type `python analysis.py`.
It prints the main results in the terminal. When `app.py` imports the file, that block is skipped.

### `charts.py` — the figures

**The set-up at the top.**

```python
SURFACE = "#20243f"      # background of every chart (same as the panels on the page)
TEAL = "#5eb6d1"
CORAL = "#ee6f87"
PURPLE = "#a47df2"
REGION_COLORS = dict(zip(REGIONS, [CORAL, TEAL, PURPLE, PEACH, ORCHID, YELLOW, MINT]))
```

- Colours are written as **hex codes**: `#RRGGBB`, two digits each for red, green and blue.
- `dict(zip(names, colours))` pairs the two lists into a dictionary, so each region always has
  the same colour on every chart, whatever the filters are.
- `LinearSegmentedColormap.from_list("heat", [...])` builds a **colormap**: a smooth ramp through
  the listed colours, used wherever colour stands for a number (heat maps, the pixel map).
- `font_manager.fontManager.addfont(...)` registers the Josefin Sans file so the charts use the
  same typeface as the page.
- `plt.rcParams.update({...})` sets Matplotlib's defaults once (background, text colour, grid,
  font sizes, no top and right border), so the individual chart functions stay short.

**The small helpers.**

| Helper | What it does |
|---|---|
| `_fig(w, h)` | `plt.subplots(figsize=(w, h), layout="constrained")`: a figure and one axes of a given size in inches |
| `_fade(ax, bars, color, direction)` | fills each bar with a gradient from the background to its colour, using `ax.imshow` |
| `_message(text)` | an empty figure with a sentence, shown when there is nothing to plot |
| `_text_on(color)` | black or white, whichever is readable on that fill (from the colour's brightness) |
| `_wrap(label)` | breaks a long axis label onto two lines |
| `_sample(df, n)` | a random sample of rows, so scatter plots draw 3,000 points instead of 102,256 |

A leading underscore in a name is a Python convention for "used only inside this file".

**Every chart function has the same shape.**

```python
def columns(values, ylabel, xlabel="", color=None, fmt="{:,.0f}", label_bars=True):
    """Vertical bar plot."""
    fig, ax = _fig()                                              # 1. make the figure
    bars = ax.bar(values.index.astype(str), values.to_numpy(), width=0.55)   # 2. draw
    if label_bars:
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=3, color=INK, fontsize=10)
    ax.set_ylabel(ylabel)                                         # 3. label
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.14)
    _fade(ax, bars, color or TEAL, "up")                          # 4. gradient fill
    return fig                                                    # 5. hand the figure back
```

**What each chart function draws, and the Matplotlib call at its centre.**

| Function | Draws | Key call | Used on |
|---|---|---|---|
| `hbar` | horizontal bars | `ax.barh` | Dataset |
| `columns` | vertical bars | `ax.bar` | Dataset, Plot gallery |
| `monthly_lines_by_region` | one line per region | `ax.plot` in a loop | Plot gallery |
| `histogram` | histogram with optional marker lines | `ax.hist`, `ax.axvline` | Central tendency, Plot gallery |
| `frequency_bars` | class-interval bars | `ax.bar` | Central tendency |
| `box_by_group` | one box per group | `ax.boxplot` | Central tendency, Plot gallery |
| `pie` | exploded donut | `ax.pie(..., wedgeprops=dict(width=...))` | Overview, Plot gallery |
| `scatter_heatwave` | scatter, heatwave days highlighted | `ax.scatter` twice | Plot gallery |
| `corr_heatmap` | the correlation matrix | `ax.imshow`, `ax.text` per cell | Correlation |
| `corr_bars` | bars left and right of zero | `ax.barh`, `ax.axvline(0)` | Correlation |
| `scatter_fit` | scatter with the regression line | `ax.scatter`, `ax.plot` | Regression |
| `correlation_examples` | three scatters side by side | `plt.subplots(1, 3)` | Correlation |
| `actual_vs_predicted` | two scatters with a diagonal | `plt.subplots(1, 2, sharex=True, sharey=True)` | Regression |
| `error_bars` | one bar per method | `ax.bar` | Regression |
| `normalization_histograms` | four small histograms | `plt.subplots(1, 4)` | Normalization |
| `scale_comparison` | box plots before and after z-score | `ax.boxplot` twice | Normalization |
| `elbow` | WCSS against k | `ax.plot(..., marker="o")` | Normalization |
| `kmeans_bins` | histogram coloured by bin | `ax.hist(list_of_arrays, stacked=True)` | Normalization |
| `pixel_map` | one pixel per record | `df.pivot_table`, `ax.imshow` | Visualization |
| `month_region_heatmap` | region × month grid | `ax.imshow`, `ax.text` | Visualization |
| `scatter_3d` | 3D scatter | `fig.add_subplot(projection="3d")` | Visualization |
| `glyph_map` | one circle per city | `ax.scatter(lon, lat, s=size, c=value)` | Visualization |
| `icon_array` | 100 dots per region | `ax.scatter` on a 10 × 10 grid | Visualization |
| `treemap` | nested rectangles | `Rectangle` patches placed by `_split` | Visualization |
| `city_dendrogram` | clustering tree | SciPy `linkage` + `dendrogram` | Visualization |

**How the gradient bars work (`_fade`).** Matplotlib bars are one flat colour. To get a fade:

1. Make a row of 256 numbers running from 0 to 1: `np.linspace(0, 1, 256)`.
2. Build a two-colour colormap from the background colour to the bar's colour.
3. Draw that row of numbers as an image exactly where the bar is, with
   `ax.imshow(ramp, extent=[left, right, bottom, top])`.
4. Hide the original bar (`bar.set_visible(False)`).

`imshow` changes the axis limits as a side effect, so the function remembers them first and puts
them back at the end.

### `app.py` — the page

The file is read top to bottom, in numbered blocks.

**Before block 1: imports and page set-up.**

```python
st.set_page_config(page_title="Heatwave Intelligence", page_icon="🌡️", layout="wide")
```

This must be the first Streamlit command. `layout="wide"` uses the full browser width.

**Block 1 — the look.**

```python
css = Path(__file__).with_name("style.css").read_text()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
```

- `Path(__file__)` is the location of `app.py`; `.with_name("style.css")` is the file beside it.
- The CSS text is wrapped in a `<style>` tag and written into the page. `unsafe_allow_html=True`
  tells Streamlit to treat the text as real HTML instead of printing it.

**Block 2 — small helpers.** Most of them build a string of HTML and write it with `html(...)`.
The important one is `chart_card`:

```python
@st.fragment
def chart_card(title, plot, *args, note="", read="", maths=None, **kwargs):
    name = card_name(title)                                        # 1
    show_code = st.session_state.get("code-" + name, False)        # 2

    with st.container(border=True, key=f"card-{name}{'--flip' if show_code else ''}"):   # 3
        with st.container(horizontal=True, horizontal_alignment="distribute", key="head-" + name):
            card_title(title, note)                                # 4
            st.toggle("Code", key="code-" + name)

        if show_code:                                              # 5
            st.code(inspect.getsource(plot), language="python")
        else:                                                      # 6
            figure = plot(*args, **kwargs)
            if figure is not None:
                st.pyplot(figure)
                plt.close(figure)
            if read:
                reading(read)
```

1. `card_name` turns "Overview" + "Heatwave Sources" into `overview-heatwave-sources`, a name
   that is safe to use in CSS.
2. Every widget with a `key` stores its value in `st.session_state`. This line reads whether this
   card's Code switch is on (False the first time).
3. The panel. Its `key` becomes a CSS class (`st-key-card-...`), which is how `style.css` finds
   it. When the switch is on, `--flip` is added to the name and the flip animation plays.
4. The header row: title and note on the left, the switch on the right.
5. Switch on: show the source code of the chart function.
6. Switch off: call the chart function. If it returned a Matplotlib figure, show it and then
   close it to free memory. The live charts return nothing because they draw themselves.

Also in block 2: the four cached functions (`load_data`, `summary_table`, `kmeans`, `elbow`),
`to_excel`, and the three live-chart functions for the Overview.

**Block 3 — the sidebar.** Navigation (`st.radio`), the three filters (`st.multiselect`), the
filtering of `df`, and the two download buttons. If the filters leave fewer than 50 rows the
script shows a warning and calls `st.stop()`, because statistics on a handful of rows are
meaningless.

**Blocks 4 to 11 — one per page**, chosen by `if page == "Overview": ... elif page == ...`. Each
block follows the same pattern:

```python
elif page == "Correlation":
    hero("Experiment 4", "Correlation", "...")                     # the page heading
    matrix = an.correlation_matrix(df, columns)                    # compute with analysis.py
    html(tiles_html([...]))                                        # headline numbers
    chart_card("Correlation matrix", ch.corr_heatmap, matrix, ...) # charts from charts.py
    with card("The formula"):                                      # a plain panel
        st.latex(r"...")
```

| Block | Page | Calls in `analysis.py` | Charts |
|---|---|---|---|
| 4 | Overview | `key_insights` | three live charts, `pie` |
| 5 | Dataset & attributes | `ATTRIBUTE_TYPES` | `columns`, `hbar` |
| 6 | Central tendency | `summary_table`, `grouped_frequency` | `histogram`, `frequency_bars`, `box_by_group` |
| 7 | Correlation | `correlation_matrix`, `describe_r` | `corr_heatmap`, `corr_bars`, `correlation_examples` |
| 8 | Regression imputation | `imputation_experiment` | `scatter_fit`, `error_bars`, `actual_vs_predicted` |
| 8b | Regression simulator | `recommend_predictors`, `imputation_experiment` | `corr_bars`, `actual_vs_predicted` |
| 9 | Normalization & K-means | the three normalizations, `kmeans`, `elbow_wcss`, `cluster_intervals` | `scale_comparison`, `normalization_histograms`, `elbow`, `kmeans_bins` |
| 10 | Plot gallery | – | the six basic plots |
| 11 | Visualization techniques | `z_score_normalize` (inside the dendrogram) | `pixel_map`, `month_region_heatmap`, `scatter_3d`, `glyph_map`, `icon_array`, `treemap`, `city_dendrogram` |

### `style.css` — the look

CSS is a list of rules. Each rule is a **selector** (which elements) and **properties** (how they
look):

```css
.card-title { font-size: 1.45rem; letter-spacing: .04em; color: var(--ink); }
/* selector    property: value;                                              */
```

| CSS idea | Example from the file | What it does |
|---|---|---|
| Class selector | `.big-word` | every element written with `class="big-word"` |
| Attribute selector | `[class*="st-key-card-"]` | every element whose class *contains* that text, i.e. every panel |
| Variable | `--teal: #5eb6d1;` then `var(--teal)` | a colour defined once and reused |
| Gradient | `linear-gradient(90deg, rgba(255,255,255,0.04), var(--c))` | a fade from nearly transparent to a colour (the bars) |
| Flexbox | `display: flex; gap: 12px;` | puts children in a row (a bar and its value) |
| Grid | `grid-template-columns: repeat(4, 1fr);` | four equal columns (the insight tiles) |
| Transition | `transition: filter .2s ease;` | animates a change smoothly, e.g. on hover |
| Hover | `.bar-row:hover .bar { filter: brightness(1.25); }` | a different look while the mouse is over it |
| Keyframes | `@keyframes flip { from {...} to {...} }` | a named animation with a start and an end state |
| Media query | `@media (max-width: 1100px) { ... }` | different rules on a narrow screen |
| `!important` | `display: none !important;` | wins over Streamlit's own styling |

**The flip in detail.**

```css
[class*="st-key-card-"][class*="--flip"] { animation: flip .6s cubic-bezier(.2, .8, .25, 1) backwards; }
@keyframes flip {
  from { transform: perspective(1800px) rotateY(-90deg); opacity: .1; }
  to   { transform: perspective(1800px) rotateY(0deg);   opacity: 1; }
}
```

The first rule matches a panel only when its name contains `--flip`. The animation starts with the
panel turned edge-on (`rotateY(-90deg)`) and nearly invisible, and ends facing the viewer.
`perspective` gives the turn depth, so it looks like a card flipping instead of squashing.

### The other files

| File | What is in it |
|---|---|
| `.streamlit/config.toml` | the theme: `base = "dark"`, the purple `primaryColor` used by switches and sliders, the panel colour as `backgroundColor` (so tables and live charts sit on the same colour as the panels), and the Google Fonts link for Josefin Sans |
| `requirements.txt` | one line per library with an exact version (`streamlit==1.65.0`), so another computer or the hosting service installs the same thing |
| `.gitignore` | what Git must not upload: the virtual environment `.venv/`, Python's cache folders, and the lab write-ups |
| `README.md` | a short description of the project and how to run it |

**Hosting.** Streamlit is a program that keeps running and talks to the browser continuously, so
it needs a host that runs a server (Streamlit Community Cloud). A static host such as Vercel only
serves files and short functions, which is why the project cannot be deployed there as it is.

---

## 15. Python, Weka and RapidMiner

The Experiment 8 write-up lists Weka and RapidMiner as tools. This project uses Python instead.

| | What it is | How you work in it |
|---|---|---|
| **Python + libraries** (this project) | a programming language | you write the code |
| **Weka** | a free desktop application for data mining | you load a CSV and click through tabs: Preprocess, Classify, Cluster, Visualize |
| **RapidMiner** (Altair AI Studio) | a desktop application | you drag boxes ("operators") onto a canvas and connect them into a pipeline |

**The same task in each.**

| Task | This project | Weka | RapidMiner |
|---|---|---|---|
| Normalize | `an.min_max_normalize(values)` | Preprocess → filter → `Normalize` | `Normalize` operator |
| K-means | `an.kmeans(values, 4)` | Cluster → `SimpleKMeans` | `k-Means` operator |
| Linear regression | `an.simple_linear_regression(x, y)` | Classify → `LinearRegression` | `Linear Regression` operator |
| Scatter plot | `ax.scatter(x, y)` | Visualize tab | Results → Visualizations |

**Why Python was the right choice here.** The experiments ask for the formulas "without built-in
functions", which a click-based tool cannot show. In Python the algorithm itself is on screen (the
Code switch), and the results can be explored live with filters. The click-based tools are faster
for a first look at a dataset and need no programming.

---

## 16. Command cheat sheets

### Pandas

```python
pd.read_csv("file.csv")                      # load
df.shape, df.columns, df.dtypes              # size, column names, types
df.head(5), df.describe(), df.info()         # first rows, summary statistics, overview
df["col"], df[["a", "b"]]                    # one column, several columns
df[df["col"] > 40]                           # filter rows
df.loc[row_label, "col"], df.iloc[0, 1]      # by label, by position
df["col"].value_counts()                     # frequencies
df.groupby("g")["col"].mean()                # group and summarise
df.groupby("g").agg(n=("col", "size"), avg=("col", "mean"))
df.sort_values("col", ascending=False)       # sort
df.isnull().sum(), df.dropna(), df.fillna(0) # missing values
pd.crosstab(df["a"], df["b"])                # two-way table
df.pivot_table(index="a", columns="b", values="c", aggfunc="sum")
df.sample(n=1000, random_state=42)           # random rows
```

### NumPy

```python
np.array([1, 2, 3]); np.zeros((2, 3)); np.ones((2, 3)); np.arange(0, 50, 10)
x.sum(); x.min(); x.max(); np.sort(x); np.unique(x)
x[2]; x[1:5]; x[x > 0]                       # indexing, slicing, masking
x + 5; x * 2; (x - m) ** 2                   # element-wise arithmetic (broadcasting)
np.column_stack([a, b]); X.T; A @ B; np.linalg.inv(A)
np.mean(x); np.median(x); np.std(x)          # built-ins (used only to CHECK our own functions)
np.random.default_rng(42).choice(n, size=k, replace=False)
```

### Matplotlib

```python
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y)            # line          ax.bar(x, h)     # bar        ax.barh(y, w)  # horizontal bar
ax.scatter(x, y)         # scatter       ax.pie(v)        # pie        ax.boxplot(d)  # box
ax.hist(v, bins=30)      # histogram     ax.imshow(m)     # image / heat map
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title(""); ax.legend(); ax.grid(True)
ax.annotate("text", (x, y)); ax.text(x, y, "text"); ax.axvline(x); ax.axhline(y)
fig.colorbar(im, ax=ax); fig.savefig("out.png", dpi=180); plt.show()
```

### Formula sheet

| Topic | Formula |
|---|---|
| Mean | $\bar{x} = \frac{1}{n}\sum x_i$ |
| Variance / std | $\sigma^2 = \frac{1}{n}\sum (x_i - \bar{x})^2$, $\;\sigma = \sqrt{\sigma^2}$ |
| IQR | $Q_3 - Q_1$ |
| CV | $\sigma / \bar{x} \times 100\%$ |
| Grouped mean | $\sum f_i m_i / N$ |
| Grouped median | $L + \frac{N/2 - cf}{f}\,h$ |
| Grouped mode | $L + \frac{f_1 - f_0}{2f_1 - f_0 - f_2}\,h$ |
| Correlation | $r = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum (x_i-\bar{x})^2 \sum (y_i-\bar{y})^2}}$ |
| Simple regression | $w_1 = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sum (x_i-\bar{x})^2}$, $\;w_0 = \bar{y} - w_1\bar{x}$ |
| Multiple regression | $b = (X^TX)^{-1}X^TY$ |
| RMSE / R² | $\sqrt{\frac{1}{m}\sum (y_i-\hat{y}_i)^2}$, $\;1 - \frac{\sum (y_i-\hat{y}_i)^2}{\sum (y_i-\bar{y})^2}$ |
| Min-max | $\frac{v - \min}{\max - \min}$ |
| Z-score | $\frac{v - \bar{x}}{\sigma}$ |
| Decimal scaling | $v / 10^j$ |
| K-means objective | $\sum_j \sum_{x_i \in C_j} \lVert x_i - \mu_j\rVert^2$ |
| Box-plot outliers | below $Q_1 - 1.5\,\text{IQR}$ or above $Q_3 + 1.5\,\text{IQR}$ |

---

## 17. Presenting it: a five-minute demo

A suggested order, with what to say at each step.

| Step | Do this | Say this |
|---|---|---|
| 1 | Open the **Overview** | "Five years of daily weather for 56 US cities. A heatwave day is one where the maximum reaches 40 °C. Only 2.5% of city-days qualify." |
| 2 | Hover over **Hot Days Per Month** | "Every summer has a spike, and the spikes are getting taller: 363 heatwave days in 2019, 701 in 2023." |
| 3 | Point at **Heatwave Sources** and **Heatwaves by City** | "60% of them are in one region, the Southwest Desert, and 25 of the 56 cities never had one." |
| 4 | Pick **Southwest Desert** in the Region filter, then clear it | "The filters apply to every page. They change one table, and every chart is drawn from that table." |
| 5 | Open **Central tendency**, flip a card with **Code** | "The statistics are written from the formulas, not library functions. Here is the variance: the mean of the squared deviations." |
| 6 | Show **Check: our functions against NumPy** | "And they match NumPy to four decimals." |
| 7 | Open **Correlation** | "Temperature rises with sunshine, r = +0.66, and falls with humidity, r = −0.44." |
| 8 | Open **Regression imputation** | "We hid 10% of the temperatures and predicted them. Multiple regression cuts the error by 32% compared with filling in the mean." |
| 9 | Open **Normalization & K-means**, move the **k** slider | "K-means chooses the bin edges itself. The elbow suggests about four bins." |
| 10 | Open **Visualization techniques** | "One example from each family in Experiment 8: pixel-oriented, geometric, icon-based and hierarchical." |

**If something goes wrong.** If the hosted link is slow to wake up, run it locally with
`streamlit run app.py`. If the page shows an error after a filter, clear the filters.

---

## 18. Likely viva questions

**Why did you write the statistics yourself instead of using NumPy functions?**
The experiments require it. Writing the formula shows we understand it; the dashboard also has a
"Check: our functions vs NumPy" table proving the hand-written results match.

**Your dataset has no missing values. How can you demonstrate imputation?**
We hide 10% of one attribute at random, predict it, and compare with the true values. That is the
only way to measure how accurate an imputation method is.

**Why is the mean of max temperature lower than its median?**
The distribution is negatively skewed: a tail of very cold winter days in the northern cities
pulls the mean down, while the median is unaffected by how extreme those days are.

**Is temperature in °C interval or ratio? Why?**
Interval. 0 °C does not mean "no temperature", so ratios are meaningless (20 °C is not twice as
hot as 10 °C). In Kelvin it would be ratio.

**What does r = −0.44 between humidity and temperature mean?**
A moderate negative linear relationship: hotter days tend to be drier. It does not prove that one
causes the other.

**What is the difference between correlation and regression?**
Correlation gives one number for the strength and direction of a linear relationship and treats
both variables the same. Regression gives an equation that predicts one variable from the other.

**Why does multiple regression beat simple regression here?**
Latitude, humidity and wind each carry information about temperature that solar radiation alone
does not, so R² rises from 0.44 to 0.54.

**How do you choose between linear and non-linear regression?**
Look at the scatter plot and the residuals. If the points follow a straight band and the errors
show no pattern, linear is enough. If the cloud is curved, or the errors are systematically
positive in one range and negative in another, a non-linear model is needed.

**What happens if data is not normalized?**
The attribute with the largest numbers dominates any distance-based method. In the dendrogram,
pressure (≈ 95) would outweigh wind speed (≈ 2) purely because of its units.

**Which normalization would you use and when?**
Min-max when a fixed range is needed and there are no extreme outliers; z-score when there are
outliers or the method assumes centred data; decimal scaling when a quick, easily reversed
rescaling is enough.

**How does K-means decide the bins, and how did you pick k?**
It alternates between assigning points to the nearest centroid and moving each centroid to the
mean of its points, which minimises the within-cluster sum of squares. We picked k from the elbow
of the WCSS curve.

**Why do the K-means bins not match the Heat_Severity_Class?**
The classes are fixed thresholds on the hot tail only. K-means is unsupervised and splits the
whole range into groups of similar size and spread.

**What is the difference between a bar plot and a histogram?**
A bar plot compares separate categories; a histogram shows the distribution of one continuous
variable by counting values in adjacent bins.

**When is a pie chart acceptable?**
For parts of one whole with only a few slices and a clear difference between them. That is why the
small regions are grouped into "Other".

**What is the difference between a chatbot and a dashboard?**
A dashboard is a fixed visual layout where the user explores by looking and filtering. A chatbot
answers questions in conversation, one at a time, in natural language.

**What role does visualization play in data mining?**
It helps at every stage: spotting outliers and skew before modelling, choosing attributes (the
correlation matrix), checking a model (predicted-vs-actual plot), and explaining results to people
who will not read a table.

**How does a filter in the sidebar reach every chart?**
It does not reach them one by one. The filter changes the table `df`, Streamlit re-runs the
script, and every chart is drawn again from the new `df`.

**What happens when you flip a card with the Code switch?**
The switch stores True in Streamlit's session state. The card function then shows
`inspect.getsource(plot)`, the text of the chart function, instead of calling it. The turning
motion is a CSS animation.

**Which charts are interactive, and how?**
Three on the Overview use `st.area_chart` and `st.bar_chart`, which are built into Streamlit, so
they show values on hover and can be zoomed. All other charts are Matplotlib figures.

**Why are the heatwave days per region drawn with fixed colours?**
So a region keeps the same colour on every chart and under every filter. Colour identifies the
region; it must not depend on which rows happen to be selected.

**Why sample 3,000 points for the scatter plots?**
Drawing all 102,256 points would be slow and the dots would overlap into a solid block. A random
sample shows the same pattern. The statistics (r, the regression line) still use every row.

**Why did you choose Python instead of Weka or RapidMiner?**
The experiments ask for the formulas without built-in functions, which a click-based tool cannot
show. In Python the algorithm is visible, and a dashboard lets the results be explored live.

---

## 19. Glossary

| Term | Meaning |
|---|---|
| **Attribute** | a column of the dataset |
| **Record** | a row of the dataset (here: one city on one day) |
| **DataFrame** | Pandas' table of rows and columns |
| **Series** | one column of a DataFrame, or any labelled list of values |
| **Array** | NumPy's list of numbers that supports arithmetic on all elements at once |
| **Mask** | a column of True/False values used to pick rows |
| **Vectorisation** | doing arithmetic on a whole array in one step instead of a loop |
| **Broadcasting** | NumPy stretching a smaller shape to match a larger one in arithmetic |
| **Central tendency** | where the middle of the data is: mean, median, mode |
| **Variability / dispersion** | how spread out the data is: range, variance, standard deviation, IQR |
| **Skew** | a distribution with a longer tail on one side |
| **Quartile** | one of the three values that cut sorted data into four equal parts |
| **Outlier** | a value far from the rest, beyond 1.5 × IQR from the box in a box plot |
| **Correlation** | how strongly two attributes move together in a straight line |
| **Regression** | fitting an equation that predicts one attribute from others |
| **Least squares** | choosing the line that makes the sum of squared errors smallest |
| **Coefficient** | a number multiplying a predictor in a regression equation |
| **Intercept** | the predicted value when every predictor is zero |
| **Residual / error** | actual value minus predicted value |
| **Imputation** | filling in missing values |
| **Normalization** | rescaling attributes to a common range |
| **Discretization** | replacing continuous values with a small number of bins |
| **Cluster** | a group of similar data points |
| **Centroid** | the mean of the points in a cluster |
| **WCSS** | within-cluster sum of squares: what K-means minimises |
| **Unsupervised** | a method that uses no labels (K-means, hierarchical clustering) |
| **Colormap** | a function from a number to a colour |
| **Figure / Axes** | Matplotlib's whole image / one plot inside it |
| **Widget** | an input on the page: a filter, slider, switch or button |
| **Re-run** | Streamlit executing the script again after a widget changes |
| **Cache** | a stored result that is reused instead of being recomputed |
| **Session state** | values Streamlit remembers between re-runs |
| **CSS** | the language that describes how a web page looks |
| **Repository** | a project folder tracked by Git (and stored on GitHub) |
