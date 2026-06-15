# Generuje wykresy (stonowana paleta) + results.json na potrzeby prezentacji
import json, warnings, matplotlib
matplotlib.use('Agg'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, IsolationForest)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error, r2_score)
from sklearn.base import clone
import shap, os

# ---------- STONOWANA PALETA / STYL ----------
MUTED = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD', '#937860', '#8C8C8C']
INK = '#2F3B4C'; BG = '#F7F5F1'; GRID = '#D9D4CC'
sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
    'axes.edgecolor': GRID, 'grid.color': GRID, 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'axes.titlecolor': INK, 'font.size': 12, 'axes.titleweight': 'bold',
})
A = os.path.join(os.path.dirname(__file__), 'prezentacja_assets')
os.makedirs(A, exist_ok=True)
base = os.path.dirname(__file__)
R = {}  # wyniki do prezentacji

# ---------- DANE ----------
train_df = pd.read_csv(f'{base}/train_set.csv', index_col='Id')
test_df = pd.read_csv(f'{base}/test_set.csv', index_col='Id')
R['n_train_raw'] = len(train_df); R['n_test'] = len(test_df)
R['n_num'] = len(train_df.select_dtypes('number').columns) - 1
R['n_cat'] = len(train_df.select_dtypes('object').columns)
R['skew_raw'] = float(train_df['SalePrice'].skew())
R['price_mean'] = float(train_df['SalePrice'].mean())

# Wykres 1: rozkład SalePrice (orig + log)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].hist(train_df['SalePrice'], bins=50, color=MUTED[0], edgecolor='white', alpha=0.9)
ax[0].set_title(f'Rozkład SalePrice (skośność = {R["skew_raw"]:.2f})'); ax[0].set_xlabel('Cena ($)'); ax[0].set_ylabel('Liczba domów')
ax[1].hist(np.log1p(train_df['SalePrice']), bins=50, color=MUTED[1], edgecolor='white', alpha=0.9)
ax[1].set_title(f'log(1+SalePrice) (skośność = {np.log1p(train_df["SalePrice"]).skew():.2f})'); ax[1].set_xlabel('log(1 + Cena)')
plt.tight_layout(); plt.savefig(f'{A}/01_dist.png', dpi=150); plt.close()

# ---------- IMPUTACJA DZIEDZINOWA ----------
none_cat = ['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
            'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
zero_num = ['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath',
            'GarageYrBlt','GarageCars','GarageArea']
def fill_struct(df):
    df = df.copy()
    for c in none_cat:
        if c in df: df[c] = df[c].astype('object').fillna('None')
    for c in zero_num:
        if c in df: df[c] = df[c].fillna(0)
    return df
miss_before = int(train_df.isnull().sum().sum())
train_df = fill_struct(train_df); test_df = fill_struct(test_df)
R['miss_before'] = miss_before; R['miss_after'] = int(train_df.isnull().sum().sum())

# ---------- FEATURE ENGINEERING ----------
def feat(df):
    df = df.copy()
    df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']
    df['TotalFlrSF'] = df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath']
    df['TotalBsmtBath'] = df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
    df['TotalBathTotal'] = df['TotalBath'] + df['TotalBsmtBath']
    df['HouseAge'] = (df['YrSold']-df['YearBuilt']).clip(lower=0)
    df['RemodAge'] = (df['YrSold']-df['YearRemodAdd']).clip(lower=0)
    df['GarageAreaPerCar'] = df['GarageArea']/(df['GarageCars']+1)
    df['BedroomToKitchenRatio'] = df['BedroomAbvGr']/(df['KitchenAbvGr']+1)
    df['RoomSize'] = df['GrLivArea']/(df['TotRmsAbvGrd']+1)
    df['QualityCondition'] = df['OverallQual']*df['OverallCond']
    out = [c for c in df.columns if 'Porch' in c or 'Deck' in c]
    df['TotalOutdoorSF'] = df[out].sum(axis=1)
    return df
train_df = feat(train_df); test_df = feat(test_df)

# Wykres 2: korelacje z SalePrice (top 12)
num = [c for c in train_df.select_dtypes('number').columns if c != 'SalePrice']
corr = train_df[num+['SalePrice']].corr()['SalePrice'].drop('SalePrice').sort_values(ascending=False)
top = corr.head(12)
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.barh(top.index[::-1], top.values[::-1], color=MUTED[0], alpha=0.9, edgecolor='white')
ax.set_title('Top 12 cech skorelowanych z ceną (Pearson)'); ax.set_xlabel('Korelacja')
plt.tight_layout(); plt.savefig(f'{A}/02_corr.png', dpi=150); plt.close()

# ---------- OUTLIERY ----------
of = [c for c in ['GrLivArea','TotalSF','LotArea','SalePrice','TotalBsmtSF','1stFlrSF','GarageArea'] if c in train_df]
iso = IsolationForest(contamination=0.01, random_state=42)
flags = iso.fit_predict(train_df[of].fillna(train_df[of].median()))
mask = flags == -1
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.scatter(train_df.loc[~mask,'GrLivArea'], train_df.loc[~mask,'SalePrice'], s=22, color=MUTED[0], alpha=0.55, label='Normalne')
ax.scatter(train_df.loc[mask,'GrLivArea'], train_df.loc[mask,'SalePrice'], s=55, color=MUTED[2], edgecolor=INK, label='Outlier (Isolation Forest)')
ax.set_title('Detekcja wartości odstających'); ax.set_xlabel('GrLivArea'); ax.set_ylabel('SalePrice'); ax.legend()
plt.tight_layout(); plt.savefig(f'{A}/03_outliers.png', dpi=150); plt.close()
R['n_outliers'] = int(mask.sum())
train_df = train_df.loc[~mask].copy()
R['n_train_clean'] = len(train_df)

# ---------- SELEKCJA CECH ----------
from statsmodels.stats.outliers_influence import variance_inflation_factor
num = [c for c in train_df.select_dtypes('number').columns if c != 'SalePrice']
cat = train_df.select_dtypes('object').columns.tolist()
corr = train_df[num+['SalePrice']].corr()['SalePrice'].sort_values(ascending=False)
topn = list(corr[1:21].index)
vif_in = train_df[topn].fillna(train_df[topn].median())
vif = pd.DataFrame({'f': topn, 'v': [variance_inflation_factor(vif_in.values, i) for i in range(len(topn))]})
high_vif = vif[vif['v'] > 10]['f'].tolist()
cp = abs(corr[1:])
sel_num = [f for f in cp[cp > 0.1].index if f not in high_vif]
sel_cat = cat[:10]
R['n_sel_num'] = len(sel_num); R['n_sel_cat'] = len(sel_cat)

# ---------- MODELOWANIE ----------
X = train_df[sel_num+sel_cat].copy(); Xtest = test_df[sel_num+sel_cat].copy()
y_orig = train_df['SalePrice'].copy(); y = np.log1p(y_orig)
Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=42)
yval_o = np.expm1(yval)
num_s = Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())])
num_t = Pipeline([('i', SimpleImputer(strategy='median'))])
cat_t = Pipeline([('i', SimpleImputer(strategy='most_frequent')), ('o', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20))])
pre_lin = ColumnTransformer([('num', num_s, sel_num), ('cat', cat_t, sel_cat)])
pre_tree = ColumnTransformer([('num', num_t, sel_num), ('cat', cat_t, sel_cat)])
def build(est, pca):
    s = [('preprocessor', pre_lin if pca else pre_tree)]
    if pca: s.append(('pca', PCA(n_components=0.95)))
    s.append(('model', est)); return Pipeline(s)
cv = KFold(5, shuffle=True, random_state=42)
specs = {
 'Linear Regression': (LinearRegression(), {}, True, None),
 'Ridge Regression': (Ridge(), {'model__alpha':[0.1,1,10,100,1000]}, True, 'grid'),
 'Lasso Regression': (Lasso(max_iter=10000), {'model__alpha':[0.0005,0.001,0.01,0.1,1]}, True, 'grid'),
 'K-Nearest Neighbors': (KNeighborsRegressor(), {'model__n_neighbors':[3,5,7,10,15],'model__weights':['distance','uniform']}, True, 'grid'),
 'Random Forest': (RandomForestRegressor(random_state=42, n_jobs=-1), {'model__n_estimators':[100,200,300],'model__max_depth':[15,20,25]}, False, 'rand'),
 'Extra Trees': (ExtraTreesRegressor(random_state=42, n_jobs=-1), {'model__n_estimators':[100,200,300],'model__max_depth':[15,20,25]}, False, 'rand'),
 'Gradient Boosting': (GradientBoostingRegressor(random_state=42), {'model__n_estimators':[100,200,300],'model__learning_rate':[0.01,0.05,0.1]}, False, 'rand'),
 'XGBoost': (XGBRegressor(random_state=42, verbosity=0), {'model__n_estimators':[100,200,300],'model__learning_rate':[0.01,0.05,0.1],'model__max_depth':[3,5,7]}, False, 'rand'),
}
tuned = {}; rows = []
for name,(est,grid,pca,kind) in specs.items():
    pipe = build(est, pca)
    if kind == 'grid':
        s = GridSearchCV(pipe, grid, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1); s.fit(Xtr, ytr); best = s.best_estimator_; cvr = -s.best_score_
    elif kind == 'rand':
        s = RandomizedSearchCV(pipe, grid, cv=cv, n_iter=10, scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42); s.fit(Xtr, ytr); best = s.best_estimator_; cvr = -s.best_score_
    else:
        sc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1); pipe.fit(Xtr, ytr); best = pipe; cvr = -sc.mean()
    tuned[name] = best
    p = np.expm1(best.predict(Xval))
    rows.append({'Model':name,'R2':r2_score(yval_o,p),'RMSE':np.sqrt(mean_squared_error(yval_o,p)),
                 'MAE':mean_absolute_error(yval_o,p),'MAPE':mean_absolute_percentage_error(yval_o,p),'CV':cvr})
res = pd.DataFrame(rows).sort_values('R2', ascending=False).reset_index(drop=True)
R['ranking'] = res.round(4).to_dict('records')
R['pca_dims'] = int(tuned['Ridge Regression'].named_steps['pca'].n_components_)
R['n_feat_tree'] = int(tuned['Random Forest'].named_steps['preprocessor'].transform(Xtr).shape[1])
best_name = res.iloc[0]['Model']; R['best'] = best_name

# Wykres 4: ranking modeli (R2 + RMSE)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
rr = res.sort_values('R2')
ax[0].barh(rr['Model'], rr['R2'], color=MUTED[1], alpha=0.9, edgecolor='white')
ax[0].set_title('R² (zbiór walidacyjny)'); ax[0].set_xlabel('R²'); ax[0].set_xlim(0.6, 0.9)
rm = res.sort_values('RMSE', ascending=False)
ax[1].barh(rm['Model'], rm['RMSE'], color=MUTED[2], alpha=0.9, edgecolor='white')
ax[1].set_title('RMSE (zbiór walidacyjny)'); ax[1].set_xlabel('RMSE ($)')
plt.tight_layout(); plt.savefig(f'{A}/04_ranking.png', dpi=150); plt.close()

# ---------- REZYDUA ----------
bm = tuned[best_name]; vp = np.expm1(bm.predict(Xval)); resid = yval_o.values - vp
R['mae_val'] = float(np.abs(resid).mean()); R['resid_bias'] = float(resid.mean())
R['mape_med'] = float(np.median(np.abs(resid/yval_o.values*100)))
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(yval_o, vp, s=22, color=MUTED[0], alpha=0.55)
lim = [yval_o.min(), yval_o.max()]; ax[0].plot(lim, lim, '--', color=MUTED[2])
ax[0].set_title(f'Predicted vs Actual — {best_name}'); ax[0].set_xlabel('Cena rzeczywista ($)'); ax[0].set_ylabel('Predykcja ($)')
ax[1].hist(resid, bins=40, color=MUTED[3], alpha=0.9, edgecolor='white'); ax[1].axvline(0, color=MUTED[2], ls='--')
ax[1].set_title('Rozkład rezyduów'); ax[1].set_xlabel('Rezyduum ($)'); ax[1].set_ylabel('Liczba domów')
plt.tight_layout(); plt.savefig(f'{A}/05_residuals.png', dpi=150); plt.close()

# ---------- SHAP ----------
pre = bm.named_steps['preprocessor']; mstep = bm.named_steps['model']
fn = pre.get_feature_names_out(); Xvt = pre.transform(Xval)
Xvdf = pd.DataFrame(Xvt, columns=fn, index=Xval.index)
expl = shap.TreeExplainer(mstep); sv = expl.shap_values(Xvdf)
imp = pd.Series(np.abs(sv).mean(axis=0), index=fn).sort_values(ascending=False)
R['shap_top'] = [(i.replace('num__','').replace('cat__',''), round(float(v),4)) for i, v in imp.head(8).items()]
plt.figure()
shap.summary_plot(sv, Xvdf, max_display=12, show=False, color_bar=True)
fig = plt.gcf(); fig.set_size_inches(11, 6); fig.set_facecolor(BG)
for a in fig.axes: a.set_facecolor(BG)
plt.title(f'SHAP — globalna istotność cech ({best_name})', color=INK, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{A}/06_shap.png', dpi=150, facecolor=BG); plt.close()

# ---------- TEST ----------
testp = pd.DataFrame(index=Xtest.index)
for name, model in tuned.items():
    testp[name] = np.expm1(clone(model).fit(X, y).predict(Xtest))
top4 = res.head(4)['Model'].tolist(); testp['Ensemble'] = testp[top4].mean(axis=1)
R['test_mean'] = float(testp[best_name].mean()); R['test_median'] = float(testp[best_name].median())
fig, ax = plt.subplots(figsize=(11, 5.2))
for i, name in enumerate(top4):
    ax.hist(testp[name], bins=45, alpha=0.45, color=MUTED[i], label=name)
ax.hist(y_orig, bins=45, alpha=0.35, color=INK, label='Trening (rzeczywiste)')
ax.set_title('Predykcje na test_set.csv vs ceny treningowe'); ax.set_xlabel('Cena ($)'); ax.set_ylabel('Liczba domów'); ax.legend()
plt.tight_layout(); plt.savefig(f'{A}/07_test.png', dpi=150); plt.close()

json.dump(R, open(f'{A}/results.json', 'w'), indent=1)
print('CHARTS + results.json OK ->', A)
print('best:', best_name, '| ranking:', [(r['Model'], round(r['R2'],3)) for r in R['ranking']])