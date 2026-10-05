# Heatwave Intelligence Dashboard — FDS Mini Project

An interactive dashboard on `heatwave_dataset_us.csv` (102,256 daily weather records,
56 US cities, 7 regions, 2019–2023). Every section applies one of the FDS lab experiments.

## Run it

```bash
cd "untitled folder 2"
source .venv/bin/activate          # first time on another machine: python3 -m venv .venv && pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at http://localhost:8501. To print the main results in the terminal instead:

```bash
python analysis.py
```

## Files

| File | What it does |
|---|---|
| `analysis.py` | All the statistics, written by hand (no `np.mean`, `np.corrcoef`, sklearn models) |
| `charts.py` | The Matplotlib figures |
| `app.py` | Streamlit page: puts everything on screen, top to bottom |
| `style.css` | How the page looks: colours, cards, the flip animation |
| `fonts/` | The Josefin Sans font file used by the charts |
| `files/` | The dataset and the lab write-ups |
| `PROJECT_GUIDE.md` | Full explanation: the maths behind every section and every Pandas / NumPy / Matplotlib command used |

## Sections and the experiment each one covers

| Section | Experiment | What is shown |
|---|---|---|
| Overview | – | KPIs, key insights, heatwave trend, regions, cities |
| Dataset & attributes | 1, 2 | Attribute classification, `head`, `describe`, filtering, `groupby` |
| Central tendency & variability | 3 | Mean, median, mode, variance, std dev, IQR, grouped frequency distribution |
| Correlation | 4 | Pearson r from the formula, correlation matrix, scatter plots |
| Regression imputation | 5 | Simple (least squares) and multiple (normal equations) regression to predict hidden values |
| Normalization & discretization | 6 | Min-max, z-score, decimal scaling; K-means binning with the elbow method |
| Plot gallery | 7 | Line, bar, scatter, pie, box and histogram plots |
| Visualization techniques | 8 | Pixel-oriented, geometric projection (3D), icon-based, hierarchical (tree map, dendrogram) |

The Year / Region / Season filters in the sidebar apply to every section.

## Controls

- **Live charts** (Overview): three charts use Streamlit's built-in charts, so you can hover to read values and scroll to zoom.
- **Code** (top right of every chart card): flips the card over to show the Matplotlib code that draws
  the chart, and the formula from `analysis.py` where there is one.
- **Download data** (bottom of the sidebar): the currently filtered records as CSV or Excel.
  The Excel file also has a summary-statistics sheet; the full dataset takes a few seconds to build.

## Deploying

Streamlit needs a server that stays running and keeps a WebSocket open, so it **cannot run on
Vercel** (Vercel only runs short serverless functions). Use Streamlit Community Cloud, which is free:

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io, sign in with GitHub and choose **Create app**.
3. Pick the repository, branch `main`, main file `app.py`, and Python 3.11 or newer under *Advanced settings*.
4. Deploy. It installs `requirements.txt` and reads `.streamlit/config.toml` automatically.
