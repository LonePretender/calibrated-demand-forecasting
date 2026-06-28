import os
import pandas as pd, numpy as np
from lightgbm import LGBMRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('data', exist_ok=True)

# 1. DATA PREP
df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='latin1',
                  usecols=['order date (DateOrders)','Category Name','Order Item Quantity'])
df['order_date'] = pd.to_datetime(df['order date (DateOrders)'])
df['week'] = df['order_date'].dt.to_period('W').dt.start_time
weekly = df.groupby(['week','Category Name'])['Order Item Quantity'].sum().reset_index()
weekly.columns = ['week','category','qty']
weekly.to_csv('data/weekly_demand.csv', index=False)

weekly = pd.read_csv('data/weekly_demand.csv', parse_dates=['week'])
top_cats = weekly.groupby('category')['qty'].sum().sort_values(ascending=False).head(10).index
weekly = weekly[weekly['category'].isin(top_cats)].copy()
all_weeks = weekly['week'].sort_values().unique()
full_idx = pd.MultiIndex.from_product([all_weeks, top_cats], names=['week','category'])
weekly = weekly.set_index(['week','category']).reindex(full_idx, fill_value=0).reset_index()
weekly = weekly.sort_values(['category','week']).reset_index(drop=True)

def add_features(g):
    g = g.sort_values('week').copy()
    for lag in [1,2,4,8]:
        g[f'lag{lag}'] = g['qty'].shift(lag)
    g['roll4'] = g['qty'].shift(1).rolling(4).mean()
    g['roll8'] = g['qty'].shift(1).rolling(8).mean()
    g['vol8']  = g['qty'].shift(1).rolling(8).std()
    g['month'] = g['week'].dt.month
    return g

feat = pd.concat([add_features(g) for _, g in weekly.groupby('category')], ignore_index=True)
feat = feat.dropna().reset_index(drop=True)
feat['cat_code'] = feat['category'].astype('category').cat.codes
feature_cols = ['lag1','lag2','lag4','lag8','roll4','roll8','month','cat_code']

weeks_sorted = sorted(feat['week'].unique())
n = len(weeks_sorted)
train_weeks = weeks_sorted[:n-42]
calib_weeks = weeks_sorted[n-42:n-21]
test_weeks  = weeks_sorted[n-21:]
train = feat[feat['week'].isin(train_weeks)].copy()
calib = feat[feat['week'].isin(calib_weeks)].copy()
test  = feat[feat['week'].isin(test_weeks)].copy()

# 2. METHOD A — QUANTILE REGRESSION + CQR
print("=" * 55)
print("METHOD A: Quantile Regression + CQR calibration")
print("=" * 55)

def fit_q(alpha):
    m = LGBMRegressor(objective='quantile', alpha=alpha, n_estimators=200, max_depth=4,
                       learning_rate=0.05, min_child_samples=5, verbose=-1)
    m.fit(train[feature_cols], train['qty'])
    return m

m_med = fit_q(0.5)
calib['point'] = m_med.predict(calib[feature_cols])
test['point']  = m_med.predict(test[feature_cols])

levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
cqr_curve, raw_curve = [], []
for lvl in levels:
    a = (1-lvl)/2
    m_lo, m_hi = fit_q(a), fit_q(1-a)
    calib[f'lo_{lvl}'] = m_lo.predict(calib[feature_cols])
    calib[f'hi_{lvl}'] = m_hi.predict(calib[feature_cols])
    test[f'lo_{lvl}']  = m_lo.predict(test[feature_cols])
    test[f'hi_{lvl}']  = m_hi.predict(test[feature_cols])
    raw_cov = ((test['qty'] >= test[f'lo_{lvl}']) & (test['qty'] <= test[f'hi_{lvl}'])).mean()
    scores = np.maximum(calib[f'lo_{lvl}'] - calib['qty'], calib['qty'] - calib[f'hi_{lvl}'])
    qhat = np.quantile(scores, lvl)
    test[f'lo_cqr_{lvl}'] = test[f'lo_{lvl}'] - qhat
    test[f'hi_cqr_{lvl}'] = test[f'hi_{lvl}'] + qhat
    cqr_cov = ((test['qty'] >= test[f'lo_cqr_{lvl}']) & (test['qty'] <= test[f'hi_cqr_{lvl}'])).mean()
    cqr_curve.append((lvl, cqr_cov))
    raw_curve.append((lvl, raw_cov))

print(f"\n{'Level':<8} {'Raw QR':>8} {'CQR':>8}")
print("-" * 26)
for (lvl, c1), (_, c2) in zip(raw_curve, cqr_curve):
    print(f"{lvl:<8.2f} {c1:>8.2f} {c2:>8.2f}")


# 3. METHOD B — BOOTSTRAP ENSEMBLE
print("\n" + "=" * 55)
print("METHOD B: Bootstrap Ensemble")
print("=" * 55)

N_BOOTSTRAP = 10   # number of models; increase to 20 for smoother intervals
rng = np.random.default_rng(42)

boot_preds_test  = []   # shape: (N_BOOTSTRAP, len(test))
boot_preds_calib = []

print(f"Training {N_BOOTSTRAP} bootstrap models ...")
for i in range(N_BOOTSTRAP):
    # Resample training rows WITH replacement (bootstrap)
    idx = rng.integers(0, len(train), size=len(train))
    boot_sample = train.iloc[idx]
    m = LGBMRegressor(objective='regression', n_estimators=200, max_depth=4,
                      learning_rate=0.05, min_child_samples=5, verbose=-1)
    m.fit(boot_sample[feature_cols], boot_sample['qty'])
    boot_preds_test.append(m.predict(test[feature_cols]))
    boot_preds_calib.append(m.predict(calib[feature_cols]))
    print(f"  model {i+1}/{N_BOOTSTRAP} done")

boot_preds_test  = np.array(boot_preds_test)   # (N, len(test))
boot_preds_calib = np.array(boot_preds_calib)  # (N, len(calib))

# Point forecast = mean of all bootstrap predictions
test['boot_point']  = boot_preds_test.mean(axis=0)
calib['boot_point'] = boot_preds_calib.mean(axis=0)

# Build intervals and measure coverage at each nominal level
boot_curve = []
print(f"\n{'Level':<8} {'Boot raw':>10} {'Boot+CQR':>10}")
print("-" * 30)
for lvl in levels:
    a = (1 - lvl) / 2
    # Raw bootstrap interval: percentiles of predictions across models
    lo_boot_test  = np.percentile(boot_preds_test,  a * 100, axis=0)
    hi_boot_test  = np.percentile(boot_preds_test, (1-a) * 100, axis=0)
    lo_boot_calib = np.percentile(boot_preds_calib,  a * 100, axis=0)
    hi_boot_calib = np.percentile(boot_preds_calib, (1-a) * 100, axis=0)

    test[f'boot_lo_{lvl}'] = lo_boot_test
    test[f'boot_hi_{lvl}'] = hi_boot_test

    raw_boot_cov = ((test['qty'] >= lo_boot_test) & (test['qty'] <= hi_boot_test)).mean()

    # CQR calibration on top of bootstrap intervals
    scores = np.maximum(lo_boot_calib - calib['qty'].values,
                        calib['qty'].values - hi_boot_calib)
    qhat_boot = np.quantile(scores, lvl)
    test[f'boot_lo_cqr_{lvl}'] = lo_boot_test  - qhat_boot
    test[f'boot_hi_cqr_{lvl}'] = hi_boot_test  + qhat_boot

    boot_cqr_cov = ((test['qty'] >= test[f'boot_lo_cqr_{lvl}']) &
                    (test['qty'] <= test[f'boot_hi_cqr_{lvl}'])).mean()
    boot_curve.append((lvl, raw_boot_cov, boot_cqr_cov))
    print(f"{lvl:<8.2f} {raw_boot_cov:>10.2f} {boot_cqr_cov:>10.2f}")


# 4. CALIBRATION CURVE — both methods on one chart
lv = [l for l, _ in cqr_curve]
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], '--', color='gray', label='Ideal (perfect calibration)')
ax.plot(lv, [c for _, c in raw_curve],  'o--', color='#888780', label='Raw quantile regression')
ax.plot(lv, [c for _, c in cqr_curve],  'o-',  color='#1D9E75', linewidth=2, label='Method A: QR + CQR')
ax.plot(lv, [c for _, _, c in boot_curve], 's-', color='#2a78d6', linewidth=2, label='Method B: Bootstrap + CQR')
ax.set_xlabel('Nominal coverage level')
ax.set_ylabel('Observed coverage')
ax.set_title('Calibration curve — Method A vs Method B')
ax.legend(fontsize=8)
ax.set_xlim(0.45, 1.0); ax.set_ylim(0.35, 1.0)
plt.tight_layout()
plt.savefig('calibration_curve.png', dpi=130)
plt.close()
print("\nSaved: calibration_curve.png")


# 5. INTERVAL WIDTH COMPARISON — which method is sharper?
test['cqr_width_90']  = test['hi_cqr_0.9']         - test['lo_cqr_0.9']
test['boot_width_90'] = test['boot_hi_cqr_0.9']     - test['boot_lo_cqr_0.9']

print(f"\nInterval width at 90% (lower = sharper = better):")
print(f"  Method A (QR + CQR)      : {test['cqr_width_90'].mean():.1f} units avg")
print(f"  Method B (Bootstrap + CQR): {test['boot_width_90'].mean():.1f} units avg")

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
for ax, col, title, color in zip(
    axes,
    ['cqr_width_90', 'boot_width_90'],
    ['Method A: QR + CQR', 'Method B: Bootstrap + CQR'],
    ['#1D9E75', '#2a78d6']
):
    ax.hist(test[col], bins=25, color=color, alpha=0.8, edgecolor='white')
    ax.axvline(test[col].mean(), color='black', linewidth=1.2, linestyle='--')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Interval width (units)')
axes[0].set_ylabel('Frequency')
plt.suptitle('90% interval width distribution — narrower is sharper', fontsize=9)
plt.tight_layout()
plt.savefig('interval_width_comparison.png', dpi=130)
plt.close()
print("Saved: interval_width_comparison.png")


# 6. RISK TIERS  (using Method A CQR — primary method)
cat_mean = train.groupby('cat_code')['qty'].mean()
test['rel_width'] = test['cqr_width_90'] / test['cat_code'].map(cat_mean).clip(lower=1)
q1, q2 = test['rel_width'].quantile([0.33, 0.66])
def tier(w):
    if w <= q1: return 'Low'
    if w <= q2: return 'Medium'
    return 'High'
test['risk_tier'] = test['rel_width'].apply(tier)
print("\nRisk tiers:\n", test['risk_tier'].value_counts().to_string())


# 7. DECISION SIMULATION
HOLD_COST, STOCKOUT_COST = 1.0, 5.0
test['stock_naive']     = test['point']
test['stock_cqr']       = test['hi_cqr_0.9']
test['stock_bootstrap'] = test['boot_hi_cqr_0.9']

sim_rows = []
for col, name in [
    ('stock_naive',     'Naive\n(point forecast)'),
    ('stock_cqr',       'Method A\n(QR + CQR)'),
    ('stock_bootstrap', 'Method B\n(Bootstrap + CQR)'),
]:
    over     = (test[col] - test['qty']).clip(lower=0)
    under    = (test['qty'] - test[col]).clip(lower=0)
    cost     = (over * HOLD_COST + under * STOCKOUT_COST).sum()
    stockouts = (under > 0).sum()
    sim_rows.append((name, cost, stockouts))
    print(f"{name.replace(chr(10),' ')}: cost=${cost:,.0f}  stockouts={stockouts}/{len(test)}")

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
colors = ['#D85A30', '#1D9E75', '#2a78d6']
axes[0].bar([r[0] for r in sim_rows], [r[1] for r in sim_rows], color=colors)
axes[0].set_ylabel('Total cost ($)')
axes[0].set_title('Total inventory cost')
axes[0].tick_params(axis='x', labelsize=7.5)

axes[1].bar([r[0] for r in sim_rows], [r[2] for r in sim_rows], color=colors)
axes[1].set_ylabel('Stockout weeks')
axes[1].set_title('Stockout frequency')
axes[1].tick_params(axis='x', labelsize=7.5)

plt.suptitle('Decision simulation: naive vs Method A vs Method B', fontsize=9)
plt.tight_layout()
plt.savefig('decision_simulation.png', dpi=130)
plt.close()
print("Saved: decision_simulation.png")


# 8. UNDERESTIMATION AUDIT
test['vol_tier'] = pd.qcut(test['vol8'], 3, labels=['calm', 'moderate', 'volatile'])
print("\nUnderestimation audit — coverage by volatility (target 90%):")
print(f"{'Bucket':<12} {'Method A':>10} {'Method B':>10}")
print("-" * 34)
for vt in ['calm', 'moderate', 'volatile']:
    d = test[test['vol_tier'] == vt]
    cov_a = ((d['qty'] >= d['lo_cqr_0.9'])      & (d['qty'] <= d['hi_cqr_0.9'])).mean()
    cov_b = ((d['qty'] >= d['boot_lo_cqr_0.9']) & (d['qty'] <= d['boot_hi_cqr_0.9'])).mean()
    print(f"{vt:<12} {cov_a:>10.2f} {cov_b:>10.2f}")


# 9. SUMMARY TABLE 
print("\n" + "=" * 55)
print("SUMMARY: Method A (QR+CQR) vs Method B (Bootstrap+CQR)")
print("=" * 55)
print(f"{'Metric':<35} {'Method A':>10} {'Method B':>10}")
print("-" * 55)
cov_a_90 = dict(cqr_curve)[0.9]
cov_b_90 = dict((l, c) for l, _, c in boot_curve)[0.9]
print(f"{'Coverage @ 90% nominal':<35} {cov_a_90:>10.2f} {cov_b_90:>10.2f}")
print(f"{'Avg interval width (90%)':<35} {test['cqr_width_90'].mean():>10.1f} {test['boot_width_90'].mean():>10.1f}")
naive_cost = sim_rows[0][1]; cqr_cost = sim_rows[1][1]; boot_cost = sim_rows[2][1]
print(f"{'Total sim cost ($)':<35} {cqr_cost:>10,.0f} {boot_cost:>10,.0f}")
print(f"{'Stockout weeks':<35} {sim_rows[1][2]:>10} {sim_rows[2][2]:>10}")
print(f"{'Naive baseline cost ($)':<35} {naive_cost:>10,.0f}")
print("=" * 55)

test.to_csv('data/test_full.csv', index=False)
print("\nDone. Outputs: calibration_curve.png, decision_simulation.png,")
print("               interval_width_comparison.png, data/test_full.csv")
