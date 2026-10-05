# Heatwave Intelligence Dashboard — Complete Project Guide

This guide explains everything in the project: what each section of the dashboard does, the
maths behind it, and the Python (Pandas, NumPy, Matplotlib, Streamlit) commands that produce it.
Formulas are written in LaTeX; open this file in the VS Code Markdown preview (`Cmd+Shift+V`) or on
GitHub to see them rendered.

**Contents**

1. [The project in one page](#1-the-project-in-one-page)
2. [The dataset](#2-the-dataset)
3. [Experiment 1 — Attribute types](#3-experiment-1--attribute-types)
4. [Experiment 2 — Pandas and NumPy data handling](#4-experiment-2--pandas-and-numpy-data-handling)
5. [Experiment 3 — Central tendency and variability](#5-experiment-3--central-tendency-and-variability)
6. [Experiment 4 — Correlation coefficient](#6-experiment-4--correlation-coefficient)
7. [Experiment 5 — Predicting missing values with regression](#7-experiment-5--predicting-missing-values-with-regression)
8. [Experiment 6 — Normalization and K-means discretization](#8-experiment-6--normalization-and-k-means-discretization)
9. [Experiment 7 — The six Matplotlib plots](#9-experiment-7--the-six-matplotlib-plots)
10. [Experiment 8 — Visualization techniques](#10-experiment-8--visualization-techniques)
11. [How the dashboard itself works (Streamlit)](#11-how-the-dashboard-itself-works-streamlit)
12. [Command cheat sheets](#12-command-cheat-sheets)
13. [Likely viva questions](#13-likely-viva-questions)

---

## 1. The project in one page

**Goal.** Take a weather dataset and use the FDS lab techniques to answer: *where, when and under
what conditions do heatwaves happen?*

**Three files do all the work.**

| File | Role | Libraries |
|---|---|---|
| `analysis.py` | Every statistic, written out from its formula | NumPy, Pandas, `math` |
| `charts.py` | Every figure | Matplotlib (plus SciPy for the dendrogram) |
| `app.py` | The web page: puts the charts and tables on screen, top to bottom | Streamlit |
| `style.css` | How the page looks: colours, panels, spacing, the flip animation | CSS |

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

## 2. The dataset

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

## 3. Experiment 1 — Attribute types

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
  temperatures in the dashboard should be read with care (see section 5).
- Correlation and regression are only run on the numeric columns.

**In the code.** `ATTRIBUTE_TYPES` in `analysis.py` is a list of tuples, one per column. The page
turns it into a table and counts the scales:

```python
attributes = pd.DataFrame(an.ATTRIBUTE_TYPES, columns=[...])      # list of tuples -> table
attributes["Scale"].value_counts()                                 # how many of each scale
```

Result: 4 nominal, 2 ordinal, 8 interval, 5 ratio.

---

## 4. Experiment 2 — Pandas and NumPy data handling

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

## 5. Experiment 3 — Central tendency and variability

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

## 6. Experiment 4 — Correlation coefficient

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

## 7. Experiment 5 — Predicting missing values with regression

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

**When is regression imputation appropriate?** When the attribute with gaps is numeric, the
relationship with the predictors is roughly linear (check the scatter plot), the correlation is
reasonably strong, and the predictors themselves are not missing in those rows.

---

## 8. Experiment 6 — Normalization and K-means discretization

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

## 9. Experiment 7 — The six Matplotlib plots

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
| **Line** | `ax.plot(x, y)` | how does it change over an ordered axis? | monthly mean per region; heatwave days per month |
| **Bar** | `ax.bar(x, h)` / `ax.barh(y, w)` | how do categories compare? | heatwave days per year / region / city |
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

## 10. Experiment 8 — Visualization techniques

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
a single-hue ramp (light → dark); the correlation matrix uses a **diverging** map (blue ↔ grey ↔
red) because its values have a meaningful middle at 0.

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

## 11. How the dashboard itself works (Streamlit)

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

## 12. Command cheat sheets

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

## 13. Likely viva questions

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
