# Dodatkowe wykresy EDA: mapa ciepła, histogramy, wykresy pudełkowe (stonowana paleta)
import warnings, matplotlib, os
matplotlib.use('Agg'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

MUTED = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD', '#937860', '#8C8C8C']
INK = '#2F3B4C'; BG = '#F7F5F1'; GRID = '#D9D4CC'
sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
    'axes.edgecolor': GRID, 'grid.color': GRID, 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'axes.titlecolor': INK, 'font.size': 12, 'axes.titleweight': 'bold',
})
base = os.path.dirname(__file__); A = f'{base}/prezentacja_assets'
train = pd.read_csv(f'{base}/train_set.csv', index_col='Id')

# cechy inżynierskie (potrzebne do mapy ciepła)
train['TotalSF'] = train['TotalBsmtSF'].fillna(0) + train['GrLivArea']
train['HouseAge'] = (train['YrSold'] - train['YearBuilt']).clip(lower=0)
train['TotalBath'] = train['FullBath'] + 0.5 * train['HalfBath']

# ---------- MAPA CIEPŁA ----------
key = ['SalePrice', 'OverallQual', 'TotalSF', 'GrLivArea', 'GarageCars', 'GarageArea',
       'TotalBsmtSF', '1stFlrSF', 'TotalBath', 'YearBuilt', 'HouseAge', 'TotRmsAbvGrd', 'LotArea']
corr = train[key].corr()
fig, ax = plt.subplots(figsize=(10.5, 8.4))
cmap = sns.diverging_palette(12, 230, s=55, l=55, as_cmap=True)  # stonowany czerwono-niebieski
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap, center=0, vmin=-1, vmax=1,
            square=True, linewidths=.6, linecolor=BG, cbar_kws={'label': 'Siła zależności', 'shrink': .8},
            annot_kws={'size': 9}, ax=ax)
ax.set_title('Mapa ciepła zależności między cechami a ceną', pad=14)
plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
plt.tight_layout(); plt.savefig(f'{A}/08_heatmap.png', dpi=150); plt.close()

# ---------- HISTOGRAMY ----------
hcols = [('GrLivArea', 'Powierzchnia mieszkalna (stopy²)'),
         ('TotalSF', 'Łączna powierzchnia (stopy²)'),
         ('LotArea', 'Powierzchnia działki (stopy²)'),
         ('YearBuilt', 'Rok budowy'),
         ('TotalBsmtSF', 'Powierzchnia piwnicy (stopy²)'),
         ('GarageArea', 'Powierzchnia garażu (stopy²)')]
fig, axes = plt.subplots(2, 3, figsize=(14, 7.4))
for ax, (c, lab) in zip(axes.flat, hcols):
    ax.hist(train[c].dropna(), bins=40, color=MUTED[0], edgecolor='white', alpha=0.9)
    ax.set_title(lab, fontsize=12)
    ax.set_ylabel('Liczba domów', fontsize=10)
fig.suptitle('Rozkłady najważniejszych cech liczbowych', fontsize=15, fontweight='bold', y=1.0)
plt.tight_layout(); plt.savefig(f'{A}/09_histograms.png', dpi=150); plt.close()

# ---------- WYKRESY PUDEŁKOWE ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# cena wg ogólnej jakości
sns.boxplot(x='OverallQual', y='SalePrice', data=train, ax=axes[0],
            color=MUTED[0], fliersize=2, linewidth=1.1)
axes[0].set_title('Cena wg ogólnej jakości domu (1–10)'); axes[0].set_xlabel('Ogólna jakość (OverallQual)')
axes[0].set_ylabel('Cena ($)')
# cena wg liczby miejsc w garażu
sns.boxplot(x='GarageCars', y='SalePrice', data=train, ax=axes[1],
            color=MUTED[1], fliersize=2, linewidth=1.1)
axes[1].set_title('Cena wg pojemności garażu (liczba aut)'); axes[1].set_xlabel('Miejsca w garażu (GarageCars)')
axes[1].set_ylabel('Cena ($)')
plt.tight_layout(); plt.savefig(f'{A}/10_boxplots.png', dpi=150); plt.close()

print('EDA charts OK: 08_heatmap, 09_histograms, 10_boxplots')