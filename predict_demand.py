import pandas as pd, numpy as np, joblib, os
from lightgbm import LGBMRegressor

# always run from the folder where this script lives
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# 0. LOCATE RAW CSV  (run once — skipped if weekly_demand.csv exists)
# ─────────────────────────────────────────────
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

WEEKLY_CSV = 'data/weekly_demand.csv'
RAW_CSV    = 'DataCoSupplyChainDataset.csv'

if not os.path.exists(WEEKLY_CSV):
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"\n✗ Could not find '{RAW_CSV}'.\n"
            f"  Please place DataCoSupplyChainDataset.csv in the same folder as this script:\n"
            f"  {os.path.abspath('.')}"
        )
    print("Building weekly_demand.csv from raw CSV …")
    df = pd.read_csv(RAW_CSV, encoding='latin1',
                     usecols=['order date (DateOrders)', 'Category Name', 'Order Item Quantity'])
    df['week'] = pd.to_datetime(df['order date (DateOrders)']).dt.to_period('W').dt.start_time
    weekly_raw = (df.groupby(['week', 'Category Name'])['Order Item Quantity']
                  .sum().reset_index())
    weekly_raw.columns = ['week', 'category', 'qty']
    weekly_raw.to_csv(WEEKLY_CSV, index=False)
    print(f"  Saved {WEEKLY_CSV}  ({len(weekly_raw):,} rows)\n")

# ─────────────────────────────────────────────
# 1. LOAD & PREP DATA
# ─────────────────────────────────────────────
weekly = pd.read_csv(WEEKLY_CSV, parse_dates=['week'])
top_cats = sorted(weekly.groupby('category')['qty'].sum()
                  .sort_values(ascending=False).head(10).index.tolist())
weekly = weekly[weekly['category'].isin(top_cats)].copy()
all_weeks = weekly['week'].sort_values().unique()
full_idx = pd.MultiIndex.from_product([all_weeks, top_cats], names=['week', 'category'])
weekly = (weekly.set_index(['week', 'category'])
          .reindex(full_idx, fill_value=0).reset_index()
          .sort_values(['category', 'week']).reset_index(drop=True))

def add_features(g):
    g = g.sort_values('week').copy()
    for lag in [1, 2, 4, 8]:
        g[f'lag{lag}'] = g['qty'].shift(lag)
    g['roll4'] = g['qty'].shift(1).rolling(4).mean()
    g['roll8'] = g['qty'].shift(1).rolling(8).mean()
    g['month'] = g['week'].dt.month
    return g

feat = pd.concat([add_features(g) for _, g in weekly.groupby('category')],
                 ignore_index=True).dropna().reset_index(drop=True)
feat['cat_code'] = feat['category'].map({c: i for i, c in enumerate(top_cats)})
feature_cols = ['lag1', 'lag2', 'lag4', 'lag8', 'roll4', 'roll8', 'month', 'cat_code']

# ─────────────────────────────────────────────
# 2. TRAIN / CALIB / TEST SPLIT  (same as demand_forecast.py)
# ─────────────────────────────────────────────
weeks_sorted = sorted(feat['week'].unique())
n = len(weeks_sorted)
train_weeks = weeks_sorted[:n - 42]
calib_weeks = weeks_sorted[n - 42:n - 21]
test_weeks  = weeks_sorted[n - 21:]

train = feat[feat['week'].isin(train_weeks)].copy()
calib = feat[feat['week'].isin(calib_weeks)].copy()

# ─────────────────────────────────────────────
# 3. TRAIN MODELS  (train split only — same as evaluation pipeline)
# ─────────────────────────────────────────────
def fit_q(alpha, data=train):
    m = LGBMRegressor(objective='quantile', alpha=alpha, n_estimators=200,
                      max_depth=4, learning_rate=0.05, min_child_samples=5, verbose=-1)
    m.fit(data[feature_cols], data['qty'])
    return m

print("Training models …")
models = {
    'point':  fit_q(0.50),
    'lo_90':  fit_q(0.05),
    'hi_90':  fit_q(0.95),
    'lo_80':  fit_q(0.10),
    'hi_80':  fit_q(0.90),
}

# ─────────────────────────────────────────────
# 4. CQR CALIBRATION  (compute qhat per confidence level on calib fold)
# ─────────────────────────────────────────────
print("Calibrating with CQR …")
CQR_LEVELS = {
    0.90: ('lo_90', 'hi_90'),
    0.80: ('lo_80', 'hi_80'),
}
qhat = {}   # {level: qhat_value}

for lvl, (lo_key, hi_key) in CQR_LEVELS.items():
    lo_pred = models[lo_key].predict(calib[feature_cols])
    hi_pred = models[hi_key].predict(calib[feature_cols])
    scores  = np.maximum(lo_pred - calib['qty'].values,
                         calib['qty'].values - hi_pred)
    qhat[lvl] = float(np.quantile(scores, lvl))
    print(f"  qhat @ {int(lvl*100)}% coverage = {qhat[lvl]:.1f} units")

# ─────────────────────────────────────────────
# 5. SAVE ARTIFACTS
# ─────────────────────────────────────────────
joblib.dump(models,   'models/demand_models.joblib')
joblib.dump(top_cats, 'models/top_categories.joblib')
joblib.dump(qhat,     'models/cqr_qhat.joblib')
weekly.to_csv('data/weekly_demand_top10.csv', index=False)
print(f"\nModels saved.  Last known week: {weekly['week'].max().date()}\n")

# ─────────────────────────────────────────────
# 6. RECURSIVE MULTI-STEP INFERENCE
# ─────────────────────────────────────────────
def predict_demand_recursive(category, target_week):
    """
    Predict demand for any future week by recursively chaining predictions.

    - For weeks within history: uses real lag values directly.
    - For future weeks beyond last known data: fills missing lags with model's
      own point-forecast predictions (recursive / multi-step forecasting).

    Returns a list of dicts, one per week from (last_known+1) up to target_week,
    plus a 'steps_ahead' field and a reliability label.
    """
    target_week = pd.Timestamp(target_week)
    last_known  = pd.Timestamp(weekly['week'].max())

    # Build an extended history dict seeded with real data
    real_hist = weekly[weekly['category'] == category].set_index('week')['qty'].to_dict()
    extended  = dict(real_hist)   # will be grown with recursive predictions

    cat_code  = top_cats.index(category)

    def get_val(week):
        return extended.get(week, np.nan)

    def build_row(week):
        lags = {f'lag{n}': get_val(week - pd.Timedelta(weeks=n)) for n in [1,2,4,8]}
        roll4 = np.nanmean([get_val(week - pd.Timedelta(weeks=i)) for i in range(1,5)])
        roll8 = np.nanmean([get_val(week - pd.Timedelta(weeks=i)) for i in range(1,9)])
        return {**lags, 'roll4': roll4, 'roll8': roll8,
                'month': week.month, 'cat_code': cat_code}

    # If target is within history, just predict directly (no recursion needed)
    if target_week <= last_known:
        row = build_row(target_week)
        X   = pd.DataFrame([row])[feature_cols]
        if X.isna().any().any():
            return None
        steps = 0
        results = [_make_result(category, target_week, X, steps)]
        return results

    # Recursively predict each week from last_known+1 up to target
    results = []
    current = last_known + pd.Timedelta(weeks=1)
    steps   = 0
    while current <= target_week:
        steps += 1
        row = build_row(current)
        X   = pd.DataFrame([row])[feature_cols]
        if X.isna().any().any():
            return None   # not enough base history (dataset too short)
        r = _make_result(category, current, X, steps)
        extended[current] = r['point_forecast']   # feed prediction back as future lag
        results.append(r)
        current += pd.Timedelta(weeks=1)

    return results


def _make_result(category, week, X, steps_ahead):
    """Run all models on feature row X and return result dict."""
    point  = float(models['point'].predict(X)[0])
    lo_90r = float(models['lo_90'].predict(X)[0])
    hi_90r = float(models['hi_90'].predict(X)[0])
    lo_80r = float(models['lo_80'].predict(X)[0])
    hi_80r = float(models['hi_80'].predict(X)[0])

    # Reliability label based on how many steps ahead
    if steps_ahead == 0:
        reliability = '✅ Historical  (real data, most accurate)'
    elif steps_ahead == 1:
        reliability = '✅ Week +1     (high confidence)'
    elif steps_ahead <= 4:
        reliability = '⚠️  Week +{:d}     (moderate confidence — some lag error)'.format(steps_ahead)
    else:
        reliability = '🔴 Week +{:d}    (lower confidence — error compounds beyond ~4 weeks)'.format(steps_ahead)

    return {
        'category':       category,
        'week':           str(week.date()),
        'steps_ahead':    steps_ahead,
        'reliability':    reliability,
        'point_forecast': round(max(point, 0), 1),
        'lo_80_cqr':      round(max(lo_80r - qhat[0.80], 0), 1),
        'hi_80_cqr':      round(max(hi_80r + qhat[0.80], 0), 1),
        'lo_90_cqr':      round(max(lo_90r - qhat[0.90], 0), 1),
        'hi_90_cqr':      round(max(hi_90r + qhat[0.90], 0), 1),
        'lo_90_raw':      round(max(lo_90r, 0), 1),
        'hi_90_raw':      round(max(hi_90r, 0), 1),
    }


# ─────────────────────────────────────────────
# 7. INTERACTIVE INPUT LOOP
# ─────────────────────────────────────────────
def print_results(results):
    if results is None:
        print("  ⚠  Not enough historical data to build features.\n"
              "     The dataset needs at least 8 weeks of history before the target week.")
        return

    # If only one result (historical lookup or 1-step), print single block
    # If multi-step, print a summary table then detail for the final week
    if len(results) == 1:
        r = results[0]
        print(f"""
  Category      : {r['category']}
  Week          : {r['week']}
  Reliability   : {r['reliability']}
  Point forecast: {r['point_forecast']} units

  ── CQR-calibrated intervals ──
  80% interval  : [{r['lo_80_cqr']}  →  {r['hi_80_cqr']}]
  90% interval  : [{r['lo_90_cqr']}  →  {r['hi_90_cqr']}]

  ✅ Safety stock (90% CQR upper bound): {r['hi_90_cqr']} units
""")
    else:
        # Multi-week table
        print(f"\n  Recursive forecast for '{results[0]['category']}' "
              f"({len(results)} weeks ahead)\n")
        print(f"  {'Week':<12} {'Steps':>5}  {'Point':>7}  {'90% CQR interval':<22}  Reliability")
        print(f"  {'─'*12} {'─'*5}  {'─'*7}  {'─'*22}  {'─'*35}")
        for r in results:
            interval = f"[{r['lo_90_cqr']}  →  {r['hi_90_cqr']}]"
            # strip emoji for table alignment
            rel_short = r['reliability'].split('(')[0].strip()
            print(f"  {r['week']:<12} {r['steps_ahead']:>5}  {r['point_forecast']:>7}  {interval:<22}  {rel_short}")

        final = results[-1]
        print(f"""
  ── Final week detail: {final['week']} ──
  Point forecast: {final['point_forecast']} units
  80% interval  : [{final['lo_80_cqr']}  →  {final['hi_80_cqr']}]
  90% interval  : [{final['lo_90_cqr']}  →  {final['hi_90_cqr']}]
  ✅ Safety stock (90% CQR upper bound): {final['hi_90_cqr']} units
  {final['reliability']}
""")


def show_categories():
    print("\n  Available categories:")
    for i, c in enumerate(top_cats, 1):
        print(f"    {i:2}. {c}")
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("  Demand Forecasting — Interactive Multi-Step Prediction")
    print("=" * 60)
    show_categories()

    last_week = pd.Timestamp(weekly['week'].max())
    print(f"  💡 Last week in dataset : {last_week.date()}")
    print(f"     You can predict ANY future date — the model chains")
    print(f"     predictions recursively to reach it.")
    print(f"     Confidence decreases beyond ~4 weeks ahead.\n")

    while True:
        print("─" * 60)
        cat_input = input("  Category (name, number 1-10, 'list', or 'q' to quit): ").strip()
        if cat_input.lower() in ('q', 'quit', 'exit'):
            print("  Goodbye!")
            break
        if cat_input.lower() == 'list':
            show_categories()
            continue

        # resolve category
        if cat_input.isdigit():
            idx = int(cat_input) - 1
            if 0 <= idx < len(top_cats):
                category = top_cats[idx]
            else:
                print(f"  ✗ Number must be between 1 and {len(top_cats)}.")
                continue
        else:
            matches = [c for c in top_cats if cat_input.lower() in c.lower()]
            if len(matches) == 1:
                category = matches[0]
            elif len(matches) > 1:
                print(f"  ✗ Ambiguous — matched: {matches}\n    Please be more specific.")
                continue
            else:
                print(f"  ✗ '{cat_input}' not found. Type 'list' to see all categories.")
                continue

        print(f"  ✓ Category : {category}")

        week_input = input("  Target week (YYYY-MM-DD, any date): ").strip()
        try:
            week_ts = pd.Timestamp(week_input)
        except Exception:
            print("  ✗ Invalid date. Use YYYY-MM-DD (e.g. 2018-06-04).")
            continue

        # snap to Monday
        if week_ts.weekday() != 0:
            week_ts = week_ts - pd.Timedelta(days=week_ts.weekday())
            print(f"  ℹ  Snapped to Monday: {week_ts.date()}")

        steps = max(0, (week_ts - last_week).days // 7)
        if steps > 0:
            print(f"  ℹ  {steps} week(s) ahead — running recursive forecast …")

        results = predict_demand_recursive(category, week_ts)
        print_results(results)