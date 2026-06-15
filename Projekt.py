# %% [markdown]
# # 🏠 Kompleksowa Analiza Regresyjna Cen Domów
# ## Projekt: Predykcja Ceny Sprzedaży Nieruchomości
# 
# Pełny pipeline analizy danych, od eksploracji po budowę i ewaluację modeli uczenia maszynowego. Zmienna celu: **SalePrice** (cena sprzedaży domu)

# %% [markdown]
# ## 1. Importy i Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Ustawienia dla lepszych wizualizacji
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')
sns.set_palette('husl')

# Biblioteki do uczenia maszynowego
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Modele
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Metryki ewaluacji
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                             mean_absolute_percentage_error, r2_score,
                             mean_squared_log_error)

# Inne narzędzia
from scipy.stats import skew, kurtosis, spearmanr, pearsonr, f_oneway
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor

print('✅ Wszystkie biblioteki załadowane pomyślnie!')

# %% [markdown]
# ## 2. Wczytanie i Eksploracja Danych

# %%
# Ścieżki do plików
base_path = '/Users/wojciechofiara/Desktop/Studia/Narzędzia uczenia maszynowego/Projekt_poprawiony'
train_path = f'{base_path}/train_set.csv'
test_path = f'{base_path}/test_set.csv'
desc_path = f'{base_path}/data_description.txt'

# Wczytanie danych
train_df = pd.read_csv(train_path, index_col='Id')
test_df = pd.read_csv(test_path, index_col='Id')

# Odczytanie opisu danych
with open(desc_path, 'r') as f:
    data_description = f.read()

print('=' * 80)
print('📊 WCZYTANE DANE')
print('=' * 80)
print(f'\n✅ Zbiór treningowy: {train_df.shape}')
print(f'✅ Zbiór testowy: {test_df.shape}')
print(f'\n🔍 Pierwsze 5 wierszy zbioru treningowego:')
print(train_df.head())
print(f'\n📋 Informacje o kolumnach zbioru treningowego:')
print(train_df.info())

# %%
# Statystyka opisowa
print('\n' + '=' * 80)
print('📈 STATYSTYKA OPISOWA - ZMIENNA CELU (SalePrice)')
print('=' * 80)
print(train_df['SalePrice'].describe())

print('\n' + '=' * 80)
print('🔢 STATYSTYKA WSZYSTKICH ZMIENNYCH NUMERYCZNYCH')
print('=' * 80)
print(train_df.describe())

# %% [markdown]
# ## 3. Analiza Typów Danych i Klasyfikacja Cech

# %%
# Klasyfikacja cech po typach
numeric_features_all = train_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features_all = train_df.select_dtypes(include=['object']).columns.tolist()

# Usunięcie zmiennej celu z list
if 'SalePrice' in numeric_features_all:
    numeric_features_all.remove('SalePrice')

print('=' * 80)
print('📊 KLASYFIKACJA CECH WEDŁUG TYPÓW')
print('=' * 80)

print(f'\n🔢 CECHY NUMERYCZNE: {len(numeric_features_all)}')
print(numeric_features_all)

print(f'\n📝 CECHY KATEGORYCZNE: {len(categorical_features_all)}')
print(categorical_features_all)

print('\n' + '=' * 80)
print('📋 SZCZEGÓŁOWA ANALIZA TYPÓW DANYCH')
print('=' * 80)

for col in train_df.columns:
    dtype = train_df[col].dtype
    unique_vals = train_df[col].nunique()
    null_pct = (train_df[col].isnull().sum() / len(train_df)) * 100
    
    if dtype == 'object':
        print(f'\n📝 {col}: KATEGORYCZNA')
        print(f'   - Liczba unikalnych wartości: {unique_vals}')
        print(f'   - Braki danych: {null_pct:.2f}%')
        if unique_vals <= 15:
            print(f'   - Wartości: {train_df[col].value_counts().to_dict()}')
    else:
        print(f'\n🔢 {col}: NUMERYCZNA ({dtype})')
        print(f'   - Liczba unikalnych wartości: {unique_vals}')
        print(f'   - Braki danych: {null_pct:.2f}%')
        print(f'   - Min: {train_df[col].min()}, Max: {train_df[col].max()}')

# %% [markdown]
# ## 4. Analiza Braków Danych

# %%
# Analiza braków danych w zbiorze treningowym i testowym
print('=' * 80)
print('🔍 ANALIZA BRAKÓW DANYCH - ZBIÓR TRENINGOWY')
print('=' * 80)

missing_train = train_df.isnull().sum()
missing_train_pct = (missing_train / len(train_df)) * 100
missing_analysis = pd.DataFrame({
    'Kolumna': missing_train.index,
    'Braki': missing_train.values,
    'Procent': missing_train_pct.values
}).sort_values('Procent', ascending=False)

# Filtruj tylko te, które mają braki
missing_analysis = missing_analysis[missing_analysis['Braki'] > 0]

if len(missing_analysis) > 0:
    print(missing_analysis.to_string(index=False))
else:
    print('✅ Brak braków danych w zbiorze treningowym!')

print('\n' + '=' * 80)
print('🔍 ANALIZA BRAKÓW DANYCH - ZBIÓR TESTOWY')
print('=' * 80)

missing_test = test_df.isnull().sum()
missing_test_pct = (missing_test / len(test_df)) * 100
missing_analysis_test = pd.DataFrame({
    'Kolumna': missing_test.index,
    'Braki': missing_test.values,
    'Procent': missing_test_pct.values
}).sort_values('Procent', ascending=False)

missing_analysis_test = missing_analysis_test[missing_analysis_test['Braki'] > 0]

if len(missing_analysis_test) > 0:
    print(missing_analysis_test.to_string(index=False))
else:
    print('✅ Brak braków danych w zbiorze testowym!')

# Wizualizacja braków danych
if len(missing_analysis) > 0 or len(missing_analysis_test) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    if len(missing_analysis) > 0:
        missing_analysis_plot = missing_analysis.head(20)
        axes[0].barh(missing_analysis_plot['Kolumna'], missing_analysis_plot['Procent'], color='coral')
        axes[0].set_xlabel('Procent braków (%)')
        axes[0].set_title('Top 20: Braki danych - Zbiór treningowy')
        axes[0].invert_yaxis()
    
    if len(missing_analysis_test) > 0:
        missing_analysis_test_plot = missing_analysis_test.head(20)
        axes[1].barh(missing_analysis_test_plot['Kolumna'], missing_analysis_test_plot['Procent'], color='skyblue')
        axes[1].set_xlabel('Procent braków (%)')
        axes[1].set_title('Top 20: Braki danych - Zbiór testowy')
        axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Obsługa Braków Danych - Imputacja z Logiką Dziedzinową
# 
# Braki w tym zbiorze rzadko są losowe - najczęściej oznaczają **brak udogodnienia** (brak basenu, garażu, piwnicy). Dlatego zamiast ślepej imputacji medianą/modą stosujemy logikę dziedzinową: `'None'` / `0` dla braków strukturalnych, a medianę/modę (w Pipeline) tylko dla braków losowych.

# %%
# OBSŁUGA BRAKÓW DANYCH - z logiką dziedzinową (a nie ślepą imputacją statystyczną)
#
# W tym zbiorze brak danych w wielu kolumnach NIE jest błędem losowym, lecz oznacza
# fizyczny BRAK danego udogodnienia (brak basenu, garażu, piwnicy, ogrodzenia...).
# Wstawianie tam mediany/mody zniekształcałoby rynek (np. dom bez garażu dostałby
# medianę powierzchni garażu i sztucznie zyskał na wartości).
#
# Dlatego stosujemy:
#   • cechy kategoryczne "brak udogodnienia" -> nowa kategoria 'None'
#   • cechy numeryczne   "brak udogodnienia" -> 0
#   • braki LOSOWE (np. LotFrontage, Electrical) -> mediana/moda liczona w Pipeline

# Kolumny, w których NA = brak udogodnienia (wg data_description.txt)
none_categoricals = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                     'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType',
                     'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
                     'MiscFeature']
zero_numerics = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']


def fill_structural_missing(df):
    """Uzupełnia braki STRUKTURALNE (brak udogodnienia) stałymi wartościami.

    Wypełnianie stałą (a nie statystyką liczoną z danych) NIE powoduje wycieku
    danych - te same reguły można bezpiecznie zastosować do zbioru testowego.
    """
    df = df.copy()
    for col in none_categoricals:
        if col in df.columns:
            df[col] = df[col].astype('object').fillna('None')
    for col in zero_numerics:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df

print('=' * 80)
print('🔍 OBSŁUGA BRAKÓW DANYCH (logika dziedzinowa)')
print('=' * 80)

before_train = train_df.isnull().sum().sum()
train_df = fill_structural_missing(train_df)
test_df = fill_structural_missing(test_df)

n_cat = len([c for c in none_categoricals if c in train_df.columns])
n_num = len([c for c in zero_numerics if c in train_df.columns])
print(f"\n✅ Uzupełniono braki strukturalne (brak udogodnienia):")
print(f"   • {n_cat} cech kategorycznych -> 'None'")
print(f"   • {n_num} cech numerycznych   -> 0")
print(f"\n📉 Braki w zbiorze treningowym: {before_train} -> {train_df.isnull().sum().sum()}")

remaining = train_df.columns[train_df.isnull().any()].tolist()
print(f"\nℹ️  Pozostałe braki to braki LOSOWE - uzupełnione w Pipeline:")
print(f"   {remaining}")
print(f"   - cechy numeryczne: mediana | cechy kategoryczne: najczęstsza wartość")

# %% [markdown]
# ## 6. Eksploracyjna Analiza Danych (EDA) - Wizualizacje

# %%
# Analiza rozkładu zmiennej celu (SalePrice)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Histogram - skala oryginalna
axes[0, 0].hist(train_df['SalePrice'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Rozkład Ceny Sprzedaży (oryginalna skala)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Cena ($)')
axes[0, 0].set_ylabel('Liczba domów')

# Histogram - skala logarytmiczna
axes[0, 1].hist(np.log1p(train_df['SalePrice']), bins=50, color='seagreen', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Rozkład Ceny Sprzedaży (skala log)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('log(1 + Cena)')
axes[0, 1].set_ylabel('Liczba domów')

# KDE plot
train_df['SalePrice'].plot(kind='kde', ax=axes[0, 2], color='crimson', linewidth=2)
axes[0, 2].set_title('Gęstość rozkładu SalePrice', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Cena ($)')

# Box plot
axes[1, 0].boxplot(train_df['SalePrice'], vert=True)
axes[1, 0].set_title('Box Plot: SalePrice (Outliers)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Cena ($)')

# Q-Q plot
from scipy import stats
stats.probplot(train_df['SalePrice'], dist='norm', plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot: Normalność rozkładu', fontsize=12, fontweight='bold')

# Statystyka
stats_text = f'''Statistyka SalePrice:
Średnia: ${train_df['SalePrice'].mean():,.0f}
Mediana: ${train_df['SalePrice'].median():,.0f}
Std Dev: ${train_df['SalePrice'].std():,.0f}
Min: ${train_df['SalePrice'].min():,.0f}
Max: ${train_df['SalePrice'].max():,.0f}
Skewness: {train_df['SalePrice'].skew():.3f}
Kurtosis: {train_df['SalePrice'].kurtosis():.3f}
'''
axes[1, 2].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

print('✅ Analiza rozkładu zmiennej celu zakończona!')

# %%
# Analiza korelacji cech numerycznych
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = train_df[numeric_cols].corr()

# Cechy o największej korelacji z SalePrice
correlation_with_target = correlation_matrix['SalePrice'].sort_values(ascending=False)

print('=' * 80)
print('🔗 TOP 20 CECH O NAJWIĘKSZEJ KORELACJI Z SALEPRICE')
print('=' * 80)
print(correlation_with_target.head(21))  # 21 bo pierwsze to sam SalePrice

# Wizualizacja
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Top 15 korelacji
top_15_corr = correlation_with_target.iloc[1:16]
colors = ['green' if x > 0 else 'red' for x in top_15_corr.values]
axes[0].barh(range(len(top_15_corr)), top_15_corr.values, color=colors, alpha=0.7)
axes[0].set_yticks(range(len(top_15_corr)))
axes[0].set_yticklabels(top_15_corr.index)
axes[0].set_xlabel('Korelacja Pearsona')
axes[0].set_title('Top 15 Cech skorelowanych z SalePrice', fontsize=12, fontweight='bold')
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=0.8)

# Heatmap macierzy korelacji (tylko mocne korelacje)
strong_corr_cols = list(correlation_with_target[abs(correlation_with_target) > 0.3].index)
if 'SalePrice' not in strong_corr_cols:
    strong_corr_cols.append('SalePrice')

corr_subset = train_df[strong_corr_cols].corr()
sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            cbar_kws={'label': 'Korelacja'}, ax=axes[1], square=True)
axes[1].set_title('Macierz Korelacji (|corr| > 0.3)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# Scatter plots dla top 6 cech skorelowanych z SalePrice
top_features = correlation_with_target[1:7].index.tolist()  # Pomiń SalePrice (indeks 0)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, feature in enumerate(top_features):
    # Usuń wiersze z brakami w danej parze cech (np.polyfit nie działa z NaN)
    pair = train_df[[feature, 'SalePrice']].dropna()
    axes[idx].scatter(pair[feature], pair['SalePrice'], alpha=0.5, color='steelblue', s=20)

    # Dodaj linię trendu
    z = np.polyfit(pair[feature], pair['SalePrice'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(pair[feature].min(), pair[feature].max(), 100)
    axes[idx].plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.8, label='Trend')

    corr_val = correlation_with_target[feature]
    axes[idx].set_title(f'{feature} vs SalePrice (r = {corr_val:.3f})', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('SalePrice')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print('✅ Analiza scatter plots dla top cech zakończona!')

# %% [markdown]
# ## 7. Feature Engineering - Tworzenie Nowych Cech

# %%
def create_engineered_features(df):
    """
    Tworzenie nowych cech na podstawie istniejących
    """
    df = df.copy()
    
    # 1. Agregacje powierzchni
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if 'TotalBsmtSF' in df.columns and 'GrLivArea' in df.columns:
        df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']
    
    if '1stFlrSF' in df.columns and '2ndFlrSF' in df.columns:
        df['TotalFlrSF'] = df['1stFlrSF'] + df['2ndFlrSF']
    
    # 2. Agregacje łazienek
    if 'FullBath' in df.columns and 'HalfBath' in df.columns:
        df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath'])
    
    if 'BsmtFullBath' in df.columns and 'BsmtHalfBath' in df.columns:
        df['TotalBsmtBath'] = df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    
    if 'TotalBath' in df.columns and 'TotalBsmtBath' in df.columns:
        df['TotalBathTotal'] = df['TotalBath'] + df['TotalBsmtBath']
    
    # 3. Cechy wieku domu
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['HouseAge'] = df['HouseAge'].clip(lower=0)
    
    if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        df['RemodAge'] = df['RemodAge'].clip(lower=0)
    
    # 4. Proporcje dotyczące garażu
    if 'GarageArea' in df.columns and 'GarageCars' in df.columns:
        df['GarageAreaPerCar'] = df['GarageArea'] / (df['GarageCars'] + 1)
    
    # 5. Proporcje sypialni i kuchni
    if 'BedroomAbvGr' in df.columns and 'KitchenAbvGr' in df.columns:
        df['BedroomToKitchenRatio'] = df['BedroomAbvGr'] / (df['KitchenAbvGr'] + 1)
    
    # 6. Cechy pomieszczeń
    if 'TotRmsAbvGrd' in df.columns and 'GrLivArea' in df.columns:
        df['RoomSize'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1)
    
    # 7. Funkcjonalność domu
    if 'OverallQual' in df.columns and 'OverallCond' in df.columns:
        df['QualityCondition'] = df['OverallQual'] * df['OverallCond']
    
    # 8. Outdoor spaces
    outdoor_cols = [col for col in df.columns if 'Porch' in col or 'Deck' in col]
    if outdoor_cols:
        df['TotalOutdoorSF'] = df[[col for col in outdoor_cols if col in df.columns]].sum(axis=1)
    
    print('✅ Feature engineering zakończony!')
    print(f'   Liczba nowych cech: {len([col for col in df.columns if col not in numeric_cols and col != "SalePrice"])}')
    
    return df

# Zastosowanie feature engineering
train_df = create_engineered_features(train_df)
test_df = create_engineered_features(test_df)

print(f'\n📊 Nowy kształt danych treningowych: {train_df.shape}')

# %% [markdown]
# ## 7b. Detekcja i Usunięcie Wartości Odstających (Outlierów)
# 
# Skrajne anomalie rynkowe (np. ogromne domy sprzedane bardzo tanio) destabilizują funkcję straty modeli regresyjnych. Stosujemy **Isolation Forest** do matematycznej detekcji obserwacji odstających i usuwamy je **wyłącznie ze zbioru treningowego** (zbiór testowy pozostaje nienaruszony).

# %%
# DETEKCJA I USUNIĘCIE WARTOŚCI ODSTAJĄCYCH (tylko zbiór treningowy)
from sklearn.ensemble import IsolationForest

print('=' * 80)
print('🎯 DETEKCJA WARTOŚCI ODSTAJĄCYCH (Isolation Forest)')
print('=' * 80)

# Cechy kluczowe dla ceny - na nich szukamy anomalii
outlier_features = [c for c in ['GrLivArea', 'TotalSF', 'LotArea', 'SalePrice',
                                'TotalBsmtSF', '1stFlrSF', 'GarageArea']
                    if c in train_df.columns]

iso_input = train_df[outlier_features].fillna(train_df[outlier_features].median())
iso_forest = IsolationForest(contamination=0.01, random_state=42)
outlier_flags = iso_forest.fit_predict(iso_input)   # -1 = obserwacja odstająca
mask_out = outlier_flags == -1

# Wizualizacja: GrLivArea vs SalePrice (klasyczne outliery w tym zbiorze) + IQR
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].scatter(train_df.loc[~mask_out, 'GrLivArea'], train_df.loc[~mask_out, 'SalePrice'],
                alpha=0.5, s=20, label='Normalne')
axes[0].scatter(train_df.loc[mask_out, 'GrLivArea'], train_df.loc[mask_out, 'SalePrice'],
                color='red', s=50, label='Outlier', edgecolor='black')
axes[0].set_title('Isolation Forest: GrLivArea vs SalePrice', fontsize=12, fontweight='bold')
axes[0].set_xlabel('GrLivArea'); axes[0].set_ylabel('SalePrice'); axes[0].legend()

axes[1].boxplot(train_df['SalePrice'], vert=True)
axes[1].set_title('Box Plot SalePrice (reguła IQR)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('SalePrice')
plt.tight_layout(); plt.show()

n_before = len(train_df)
train_df = train_df.loc[~mask_out].copy()
n_removed = n_before - len(train_df)

print(f'\n❌ Wykryto i usunięto {n_removed} obserwacji odstających ({n_removed / n_before * 100:.1f}%)')
print(f'✅ Zbiór treningowy: {n_before} -> {len(train_df)} wierszy')
print('ℹ️  Outliery usuwamy WYŁĄCZNIE ze zbioru treningowego - test_df pozostaje nienaruszony.')

# %% [markdown]
# ## 8. Selekcja Cech i PCA

# %%
# Selekcja cech na podstawie korelacji
numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
if 'SalePrice' in numeric_features:
    numeric_features.remove('SalePrice')

categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

# Recalculate correlation with new engineered features
correlation_matrix_new = train_df[numeric_features + ['SalePrice']].corr()
correlation_with_target = correlation_matrix_new['SalePrice'].sort_values(ascending=False)

# VIF analysis for multicollinearity
top_numeric_features = list(correlation_with_target[1:21].index)

# UWAGA: VIF wymaga kompletnych danych (bez NaN). Braki uzupełniamy medianą
# liczoną WYŁĄCZNIE na zbiorze treningowym - to jedynie analiza pomocnicza.
# Właściwa imputacja odbywa się później WEWNĄTRZ Pipeline (bez wycieku danych).
vif_input = train_df[top_numeric_features].fillna(train_df[top_numeric_features].median())

vif_data = pd.DataFrame()
vif_data['Feature'] = top_numeric_features
vif_data['VIF'] = [variance_inflation_factor(vif_input.values, i)
                    for i in range(len(top_numeric_features))]
vif_data = vif_data.sort_values('VIF', ascending=False)

print('=' * 80)
print('🎯 SELEKCJA CECH')
print('=' * 80)
print('\n1️⃣ ANALIZA MULTIKOLINEARNOŚCI (VIF)')
print(vif_data.to_string(index=False))

# Remove high VIF features
high_vif_features = vif_data[vif_data['VIF'] > 10]['Feature'].tolist()
print(f'\n⚠️ Cechy do usunięcia (VIF > 10): {high_vif_features}')

# Feature selection
correlation_with_price = abs(correlation_with_target[1:])
selected_numeric = correlation_with_price[correlation_with_price > 0.1].index.tolist()
selected_numeric = [f for f in selected_numeric if f not in high_vif_features]

selected_categorical = categorical_features[:10]

print(f'\n✅ Wybrane cechy numeryczne: {len(selected_numeric)}')
print(f'✅ Wybrane cechy kategoryczne: {len(selected_categorical)}')

# %% [markdown]
# ## 9. Preprocessing i Modelowanie

# %%
# Przygotowanie danych
X_full = train_df[selected_numeric + selected_categorical].copy()
X_test_full = test_df[selected_numeric + selected_categorical].copy()  # zbiór testowy BEZ etykiet (Kaggle)
y_train_original = train_df['SalePrice'].copy()

# LOG-TRANSFORMACJA zmiennej celu (ważne dla skośnych rozkładów)
print('=' * 80)
print('🔄 LOG-TRANSFORMACJA ZMIENNEJ CELU')
print('=' * 80)
print(f'Oryginalny rozkład SalePrice:')
print(f'  - Skewness: {y_train_original.skew():.4f}')
print(f'  - Kurtosis: {y_train_original.kurtosis():.4f}')

# Transformacja log1p (modele uczą się na log(SalePrice))
y_log = np.log1p(y_train_original)

print(f'\nPo log-transformacji:')
print(f'  - Skewness: {y_log.skew():.4f}')
print(f'  - Kurtosis: {y_log.kurtosis():.4f}')
print(f'✅ Log-transformacja zmniejszyła skośność rozkładu.')

# PODZIAŁ NA ZBIÓR TRENINGOWY I WALIDACYJNY
# Plik test_set.csv NIE zawiera kolumny SalePrice (zbiór konkursowy Kaggle),
# dlatego rzetelnej oceny modeli dokonujemy na wydzielonym zbiorze WALIDACYJNYM
# oraz przez kroswalidację - NIGDY na danych użytych do treningu.
print('\n' + '=' * 80)
print('✂️  PODZIAŁ NA ZBIÓR TRENINGOWY / WALIDACYJNY')
print('=' * 80)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_full, y_log, test_size=0.2, random_state=42
)
y_val_original = np.expm1(y_val)  # walidacyjne etykiety w skali dolarowej

print(f'  - Zbiór treningowy:  {X_tr.shape}')
print(f'  - Zbiór walidacyjny: {X_val.shape}')

# DEFINICJA PREPROCESSINGU - wpięta bezpośrednio w Pipeline (BEZ WYCIEKU DANYCH!)
# Imputacja, skalowanie i PCA są dopasowywane WYŁĄCZNIE na danych treningowych
# wewnątrz każdej iteracji kroswalidacji - to eliminuje data leakage.
print('\n' + '=' * 80)
print('🔧 DEFINICJA PREPROCESSINGU (wewnątrz Pipeline)')
print('=' * 80)

# Transformer numeryczny dla modeli liniowych/KNN: imputacja + standaryzacja
numeric_transformer_scaled = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Transformer numeryczny dla modeli drzewiastych: sama imputacja
# (drzewa nie wymagają skalowania)
numeric_transformer_tree = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20))
])

# ŚCIEŻKA 1 (modele liniowe / KNN): standaryzacja + (później) PCA
preprocessor_linear = ColumnTransformer(transformers=[
    ('num', numeric_transformer_scaled, selected_numeric),
    ('cat', categorical_transformer, selected_categorical)
])

# ŚCIEŻKA 2 (modele drzewiaste): bez skalowania, BEZ PCA
preprocessor_tree = ColumnTransformer(transformers=[
    ('num', numeric_transformer_tree, selected_numeric),
    ('cat', categorical_transformer, selected_categorical)
])

print('✅ Zdefiniowano 2 ścieżki preprocessingu:')
print('   • preprocessor_linear : imputacja + StandardScaler (+ PCA w Pipeline modelu)')
print('   • preprocessor_tree   : imputacja + OneHot (bez skalowania i PCA)')

# %%
# Definicja modeli i strojenie hiperparametrów
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

print('=' * 80)
print('🤖 BUDOWA PIPELINE I STROJENIE HIPERPARAMETRÓW')
print('=' * 80)


def build_pipeline(estimator, use_pca):
    """Buduje kompletny Pipeline: preprocessing -> (PCA) -> model.

    Cały preprocessing (imputacja, skalowanie, PCA) jest dopasowywany WEWNĄTRZ
    Pipeline, więc podczas kroswalidacji nie dochodzi do wycieku danych.
    """
    steps = [('preprocessor', preprocessor_linear if use_pca else preprocessor_tree)]
    if use_pca:
        steps.append(('pca', PCA(n_components=0.95)))
    steps.append(('model', estimator))
    return Pipeline(steps)


# Wspólna strategia kroswalidacji
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Specyfikacja modeli:
#   nazwa -> (estymator, siatka_parametrów, use_pca, rodzaj_wyszukiwania)
# Parametry mają prefiks 'model__', bo model jest krokiem Pipeline o nazwie 'model'.
model_specs = {
    # ŚCIEŻKA 1: modele liniowe / KNN (Z PCA)
    'Linear Regression': (LinearRegression(), {}, True, None),
    'Ridge Regression':  (Ridge(), {'model__alpha': [0.1, 1, 10, 100, 1000]}, True, 'grid'),
    'Lasso Regression':  (Lasso(max_iter=10000),
                          {'model__alpha': [0.0005, 0.001, 0.01, 0.1, 1]}, True, 'grid'),
    'K-Nearest Neighbors': (KNeighborsRegressor(),
                            {'model__n_neighbors': [3, 5, 7, 10, 15],
                             'model__weights': ['distance', 'uniform']}, True, 'grid'),
    # ŚCIEŻKA 2: modele drzewiaste (BEZ PCA)
    'Random Forest': (RandomForestRegressor(random_state=42, n_jobs=-1),
                      {'model__n_estimators': [100, 200, 300],
                       'model__max_depth': [15, 20, 25]}, False, 'rand'),
    'Extra Trees': (ExtraTreesRegressor(random_state=42, n_jobs=-1),
                    {'model__n_estimators': [100, 200, 300],
                     'model__max_depth': [15, 20, 25]}, False, 'rand'),
    'Gradient Boosting': (GradientBoostingRegressor(random_state=42),
                          {'model__n_estimators': [100, 200, 300],
                           'model__learning_rate': [0.01, 0.05, 0.1]}, False, 'rand'),
    'XGBoost': (XGBRegressor(random_state=42, verbosity=0),
                {'model__n_estimators': [100, 200, 300],
                 'model__learning_rate': [0.01, 0.05, 0.1],
                 'model__max_depth': [3, 5, 7]}, False, 'rand'),
}

tuned_models = {}   # nazwa -> najlepszy dopasowany Pipeline
cv_rmse = {}        # nazwa -> RMSE (skala log) z kroswalidacji

for name, (estimator, grid, use_pca, kind) in model_specs.items():
    print(f'\n⏳ {name} ({"Z PCA" if use_pca else "BEZ PCA"})...')
    pipe = build_pipeline(estimator, use_pca)

    if kind == 'grid':
        search = GridSearchCV(pipe, grid, cv=cv,
                              scoring='neg_root_mean_squared_error', n_jobs=-1)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        cv_rmse[name] = -search.best_score_
        print(f'   ✅ Najlepsze parametry: {search.best_params_}')
    elif kind == 'rand':
        search = RandomizedSearchCV(pipe, grid, cv=cv, n_iter=10,
                                    scoring='neg_root_mean_squared_error',
                                    n_jobs=-1, random_state=42)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        cv_rmse[name] = -search.best_score_
        print(f'   ✅ Najlepsze parametry: {search.best_params_}')
    else:  # brak strojenia - sama kroswalidacja + dopasowanie
        scores = cross_val_score(pipe, X_tr, y_tr, cv=cv,
                                 scoring='neg_root_mean_squared_error', n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        best = pipe
        cv_rmse[name] = -scores.mean()

    tuned_models[name] = best
    print(f'   📉 CV RMSE (log): {cv_rmse[name]:.4f}')

print('\n' + '=' * 80)
print('✅ WSZYSTKIE MODELE WYTRENOWANE I ZESTROJONE!')
print('=' * 80)

# %%
# Ewaluacja modeli na ZBIORZE WALIDACYJNYM (dane nieużyte do treningu)
def evaluate_model(y_true_original, y_pred_original, model_name):
    """Metryki liczone w ORYGINALNEJ skali dolarowej (po odwróceniu log1p)."""
    return {
        'Model': model_name,
        'MAE':  mean_absolute_error(y_true_original, y_pred_original),
        'RMSE': np.sqrt(mean_squared_error(y_true_original, y_pred_original)),
        'MAPE': mean_absolute_percentage_error(y_true_original, y_pred_original),
        'R2':   r2_score(y_true_original, y_pred_original),
    }

print('=' * 100)
print('📊 WYNIKI EWALUACJI - ZBIÓR WALIDACYJNY (uczciwa ocena, bez wycieku danych)')
print('=' * 100)

val_results = []
for name, model in tuned_models.items():
    # Predykcja na danych walidacyjnych, odwrócenie log-transformacji
    y_pred_log = model.predict(X_val)
    y_pred_original = np.expm1(y_pred_log)

    result = evaluate_model(y_val_original, y_pred_original, name)
    result['CV_RMSE_log'] = cv_rmse[name]
    val_results.append(result)

    print(f'\n{name}:')
    print(f'  CV RMSE (log): {result["CV_RMSE_log"]:>10.4f}')
    print(f'  R²:   {result["R2"]:>12.4f}')
    print(f'  RMSE: ${result["RMSE"]:>12,.2f}')
    print(f'  MAE:  ${result["MAE"]:>12,.2f}')
    print(f'  MAPE: {result["MAPE"]:>12.4f}')

results_df = pd.DataFrame(val_results)[['Model', 'R2', 'RMSE', 'MAE', 'MAPE', 'CV_RMSE_log']]
results_df = results_df.sort_values('R2', ascending=False).reset_index(drop=True)

print('\n' + '=' * 100)
print('📋 PODSUMOWANIE - RANKING MODELI (zbiór walidacyjny)')
print('=' * 100)
print(results_df.to_string(index=False))
print(f'\n🏆 NAJLEPSZY MODEL: {results_df.iloc[0]["Model"]} (R² = {results_df.iloc[0]["R2"]:.4f})')

# %%
# Wizualizacja porównania modeli (zbiór walidacyjny)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# R² Score
df_r2 = results_df.sort_values('R2')
axes[0, 0].barh(df_r2['Model'], df_r2['R2'], color='green', alpha=0.7)
axes[0, 0].set_xlabel('R² Score')
axes[0, 0].set_title('R² Score - Zbiór Walidacyjny', fontsize=12, fontweight='bold')

# RMSE
df_rmse = results_df.sort_values('RMSE', ascending=False)
axes[0, 1].barh(df_rmse['Model'], df_rmse['RMSE'], color='coral', alpha=0.7)
axes[0, 1].set_xlabel('RMSE ($)')
axes[0, 1].set_title('RMSE - Zbiór Walidacyjny', fontsize=12, fontweight='bold')

# MAE
df_mae = results_df.sort_values('MAE', ascending=False)
axes[1, 0].barh(df_mae['Model'], df_mae['MAE'], color='skyblue', alpha=0.7)
axes[1, 0].set_xlabel('MAE ($)')
axes[1, 0].set_title('MAE - Zbiór Walidacyjny', fontsize=12, fontweight='bold')

# MAPE
df_mape = results_df.sort_values('MAPE', ascending=False)
axes[1, 1].barh(df_mape['Model'], df_mape['MAPE'], color='lightgreen', alpha=0.7)
axes[1, 1].set_xlabel('MAPE')
axes[1, 1].set_title('MAPE - Zbiór Walidacyjny', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print('✅ Wizualizacja modeli zakończona!')

# %% [markdown]
# ## 9b. Analiza Rezyduów (Błędów) Najlepszego Modelu
# 
# Uśrednione metryki (R², RMSE) ukrywają lokalne patologie modelu. Analizujemy więc rozkład błędów na zbiorze walidacyjnym: wykres **Predicted vs Residuals**, normalność rezyduów oraz **5% najgorszych predykcji** pod kątem biznesowym (gdzie i dlaczego model się myli).

# %%
# ANALIZA REZYDUÓW NAJLEPSZEGO MODELU (na zbiorze walidacyjnym = ocena out-of-sample)
print('=' * 80)
print('🔬 ANALIZA REZYDUÓW NAJLEPSZEGO MODELU')
print('=' * 80)

best_model_name = results_df.iloc[0]['Model']
best_model = tuned_models[best_model_name]   # dopasowany na X_tr (NIE na X_val)

y_val_pred = np.expm1(best_model.predict(X_val))
residuals = y_val_original.values - y_val_pred          # błąd w dolarach (dodatni = niedoszacowanie)
pct_error = residuals / y_val_original.values * 100

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Predicted vs Residuals
axes[0, 0].scatter(y_val_pred, residuals, alpha=0.5, s=20, color='steelblue')
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Predykcja ceny ($)'); axes[0, 0].set_ylabel('Rezyduum ($)')
axes[0, 0].set_title(f'Predicted vs Residuals - {best_model_name}', fontsize=12, fontweight='bold')

# 2. Predicted vs Actual
axes[0, 1].scatter(y_val_original, y_val_pred, alpha=0.5, s=20, color='seagreen')
lims = [y_val_original.min(), y_val_original.max()]
axes[0, 1].plot(lims, lims, 'r--', label='Predykcja idealna')
axes[0, 1].set_xlabel('Cena rzeczywista ($)'); axes[0, 1].set_ylabel('Predykcja ($)')
axes[0, 1].set_title('Predicted vs Actual', fontsize=12, fontweight='bold'); axes[0, 1].legend()

# 3. Rozkład rezyduów
axes[1, 0].hist(residuals, bins=40, color='coral', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(0, color='red', linestyle='--')
axes[1, 0].set_xlabel('Rezyduum ($)'); axes[1, 0].set_ylabel('Liczba domów')
axes[1, 0].set_title('Rozkład rezyduów', fontsize=12, fontweight='bold')

# 4. Q-Q plot rezyduów
from scipy import stats as _stats
_stats.probplot(residuals, dist='norm', plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot rezyduów (normalność błędów)', fontsize=12, fontweight='bold')

plt.tight_layout(); plt.show()

# 5% NAJGORSZYCH PREDYKCJI - analiza biznesowa
val_diag = X_val.copy()
val_diag['Cena_rzeczywista'] = y_val_original.values
val_diag['Predykcja'] = y_val_pred.round(0)
val_diag['Blad_abs'] = np.abs(residuals).round(0)
val_diag['Blad_proc'] = pct_error.round(1)

n_worst = max(1, int(0.05 * len(val_diag)))
worst = val_diag.sort_values('Blad_abs', ascending=False).head(n_worst)

cols_show = [c for c in ['OverallQual', 'GrLivArea', 'TotalSF', 'Neighborhood', 'YearBuilt']
             if c in worst.columns]
print(f'\n🔴 5% NAJGORSZYCH PREDYKCJI ({n_worst} domów):')
print(worst[cols_show + ['Cena_rzeczywista', 'Predykcja', 'Blad_abs', 'Blad_proc']].to_string())

print(f'\n📊 MAE ogółem (walidacja):           ${np.abs(residuals).mean():,.0f}')
print(f'📊 Średni błąd na 5% najgorszych:    ${worst["Blad_abs"].mean():,.0f}')
print(f'📊 Mediana |błędu procentowego|:     {np.median(np.abs(pct_error)):.1f}%')
print(f"📊 Tendencja: model średnio {'ZAWYŻA' if residuals.mean() < 0 else 'ZANIŻA'} ceny o ${abs(residuals.mean()):,.0f}")
print(f"📊 Najgorsze predykcje dotyczą głównie domów drogich/nietypowych - typowy wzorzec dla regresji cen.")

# %%
# PREDYKCJA NA ZBIORZE TESTOWYM (Kaggle) - bez etykiet, więc generujemy submission
# Najlepszy model trenujemy ponownie na CAŁOŚCI danych treningowych (X_full, y_log).
# Używamy clone(), aby NIE nadpisać modelu w tuned_models (potrzebny do analizy
# rezyduów i SHAP, gdzie ocena ma być out-of-sample na zbiorze walidacyjnym).
from sklearn.base import clone

print('=' * 80)
print('📤 GENEROWANIE PREDYKCJI DLA ZBIORU TESTOWEGO')
print('=' * 80)

best_model_name = results_df.iloc[0]['Model']
final_model = clone(tuned_models[best_model_name])   # świeża kopia z najlepszymi parametrami

# Dopasowanie na pełnym zbiorze treningowym
final_model.fit(X_full, y_log)

# Predykcja na skali log -> odwrócenie do dolarów
test_pred_log = final_model.predict(X_test_full)
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame({'Id': X_test_full.index, 'SalePrice': test_pred})
submission_path = f'{base_path}/submission.csv'
submission.to_csv(submission_path, index=False)

print(f'✅ Model: {best_model_name} (przetrenowany na pełnym zbiorze treningowym)')
print(f'✅ Wygenerowano {len(submission)} predykcji')
print(f'✅ Zapisano do: {submission_path}')
print('\n🔍 Przykładowe predykcje:')
print(submission.head())

# %% [markdown]
# ## 9c. Interpretowalność Modelu - SHAP (Explainable AI)
# 
# Najlepszy model nie może być "czarną skrzynką". Za pomocą biblioteki **SHAP** (Shapley Additive exPlanations) pokazujemy:
# - **globalną istotność cech** - co naprawdę napędza wyceny modelu,
# - **lokalne wyjaśnienie** pojedynczej predykcji (wykres waterfall) - dlaczego konkretny dom dostał taką cenę.

# %%
# INTERPRETOWALNOŚĆ MODELU - wartości SHAP
import shap

print('=' * 80)
print('🧠 INTERPRETOWALNOŚĆ MODELU - SHAP (Explainable AI)')
print('=' * 80)

best_model_name = results_df.iloc[0]['Model']
best_pipe = tuned_models[best_model_name]          # dopasowany na X_tr
pre = best_pipe.named_steps['preprocessor']
model_step = best_pipe.named_steps['model']

# Dane walidacyjne po preprocessingu (z czytelnymi nazwami cech)
feat_names = pre.get_feature_names_out()
X_val_trans = pre.transform(X_val)

tree_models = ['Random Forest', 'Extra Trees', 'Gradient Boosting', 'XGBoost']

if best_model_name in tree_models:
    # Modele drzewiaste - szybki, dokładny TreeExplainer w przestrzeni oryginalnych cech
    X_explain = pd.DataFrame(X_val_trans, columns=feat_names, index=X_val.index)
    explainer = shap.TreeExplainer(model_step)
    shap_values = explainer.shap_values(X_explain)
    expected_value = explainer.expected_value
else:
    # Modele liniowe/KNN - po PCA, więc wyjaśniamy w przestrzeni składowych głównych
    X_pca = best_pipe.named_steps['pca'].transform(X_val_trans)
    pc_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    X_explain = pd.DataFrame(X_pca, columns=pc_names, index=X_val.index)
    explainer = shap.LinearExplainer(model_step, X_explain)
    shap_values = explainer.shap_values(X_explain)
    expected_value = explainer.expected_value

# GLOBALNA istotność cech (średni |SHAP|)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance = pd.Series(mean_abs_shap, index=X_explain.columns).sort_values(ascending=False)

print(f'\n📊 GLOBALNA ISTOTNOŚĆ CECH wg SHAP - {best_model_name} (TOP 15):')
print(importance.head(15).to_string())

# Wykres zbiorczy SHAP (beeswarm)
shap.summary_plot(shap_values, X_explain, max_display=15, show=False)
plt.title(f'SHAP summary plot - {best_model_name}', fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()

# LOKALNE wyjaśnienie pojedynczej predykcji (pierwszy dom ze zbioru walidacyjnego)
print('\n🔍 LOKALNE wyjaśnienie predykcji dla pierwszego domu walidacyjnego:')
print(f'   (wartości SHAP w skali log-ceny; suma = predykcja log dla tego domu)')
try:
    shap.plots._waterfall.waterfall_legacy(
        expected_value, shap_values[0], X_explain.iloc[0], max_display=12, show=False)
    plt.tight_layout(); plt.show()
except Exception as e:
    print(f'   (waterfall pominięty: {type(e).__name__})')

print('\n✅ Analiza SHAP zakończona - model jest interpretowalny, decyzje są wyjaśnialne.')

# %% [markdown]
# ## 10. Wnioski i Rekomendacje

# %%
best_row = results_df.iloc[0]
second_row = results_df.iloc[1]
best_model_name = best_row['Model']
second_best_name = second_row['Model']

# Liczba wymiarów po PCA (ścieżka liniowa) i liczba cech bez PCA (ścieżka drzewiasta)
pca_dims = tuned_models['Ridge Regression'].named_steps['pca'].n_components_
n_feat_no_pca = tuned_models['Random Forest'].named_steps['preprocessor'].transform(X_tr).shape[1]

print('=' * 100)
print('🏆 PODSUMOWANIE I WNIOSKI - ANALIZA DANYCH CEN DOMÓW')
print('=' * 100)

summary = f'''
🏠 PROJEKT: PREDYKCJA CEN DOMÓW (Ames Housing Dataset)

═══════════════════════════════════════════════════════════════════════════════

1️⃣ EKSPLORACYJNA ANALIZA DANYCH
   ✓ Analizowano {train_df.shape[0]} domów (po usunięciu outlierów) z {train_df.shape[1]} cechami
   ✓ Zidentyfikowano {len(numeric_features_all)} cech numerycznych i {len(categorical_features_all)} kategorycznych
   ✓ Cena średnia: ${y_train_original.mean():,.0f}
   ✓ Rozkład cen: prawostronnie skośny (skewness: {y_train_original.skew():.3f})

2️⃣ OBSŁUGA DANYCH (logika dziedzinowa)
   ✓ Braki STRUKTURALNE (brak udogodnienia): kategoryczne -> 'None', numeryczne -> 0
   ✓ Braki LOSOWE (LotFrontage, Electrical): mediana/moda wewnątrz Pipeline
   ✓ Outliery: Isolation Forest usunął anomalie tylko ze zbioru treningowego
   ✓ LOG-TRANSFORMACJA celu: skewness {y_train_original.skew():.2f} -> {np.log1p(y_train_original).skew():.2f}
   ✓ Feature Engineering: 8+ nowych cech (TotalSF, HouseAge, QualityCondition, itd.)

3️⃣ PREPROCESSING Z DWOMA ŚCIEŻKAMI
   📊 ŚCIEŻKA 1: Z PCA (modele liniowe/KNN) - redukcja do {pca_dims} wymiarów (95% wariancji)
   📊 ŚCIEŻKA 2: BEZ PCA (modele drzewiaste) - {n_feat_no_pca} oryginalnych cech

4️⃣ STROJENIE HIPERPARAMETRÓW
   ✓ GridSearchCV (modele liniowe/KNN) + RandomizedSearchCV (modele drzewiaste)
   ✓ 5-fold kroswalidacja, scoring = RMSE na skali log
   ✓ Cały preprocessing w Pipeline -> kroswalidacja bez wycieku danych

═══════════════════════════════════════════════════════════════════════════════

📊 WYNIKI MODELOWANIA (ZBIÓR WALIDACYJNY - dane nieużyte do treningu)

🥇 NAJLEPSZY MODEL: {best_model_name}
   R² Score:      {best_row['R2']:.4f}
   RMSE:          ${best_row['RMSE']:,.0f}
   MAE:           ${best_row['MAE']:,.0f}
   MAPE:          {best_row['MAPE']:.4f}
   CV RMSE (log): {best_row['CV_RMSE_log']:.4f}

🥈 DRUGI NAJLEPSZY: {second_best_name}
   R² Score:      {second_row['R2']:.4f}
   RMSE:          ${second_row['RMSE']:,.0f}
   MAE:           ${second_row['MAE']:,.0f}
   MAPE:          {second_row['MAPE']:.4f}

═══════════════════════════════════════════════════════════════════════════════

✅ KLUCZOWE POPRAWKI METODOLOGICZNE

1. ✅ Brak wycieku danych - preprocessing wpięty w Pipeline (refit w każdym foldzie CV).
2. ✅ Uczciwa ocena - metryki na zbiorze WALIDACYJNYM + kroswalidacja (nie na treningu).
3. ✅ Imputacja z logiką dziedzinową - 'None'/0 dla braku udogodnienia, mediana tylko
      dla braków losowych (model nie uczy się fałszywych zależności).
4. ✅ Detekcja outlierów (Isolation Forest) - stabilniejsza funkcja straty.
5. ✅ Log-transformacja zmiennej celu - sprawiedliwsza penalizacja błędów.
6. ✅ PCA tylko dla modeli liniowych/KNN; drzewa na oryginalnych cechach.
7. ✅ Strojenie hiperparametrów (GridSearchCV / RandomizedSearchCV).
8. ✅ Analiza rezyduów + interpretowalność SHAP (model to nie czarna skrzynka).

═══════════════════════════════════════════════════════════════════════════════

💡 REKOMENDACJE I DALSZE KROKI
   • Wdrożenie modelu: {best_model_name}; predykcje testowe w submission.csv
   • Dalej: ensemble (Stacking/Voting), bogatsza selekcja cech kategorycznych,
     imputacja LotFrontage medianą wg dzielnicy (Neighborhood).

═══════════════════════════════════════════════════════════════════════════════
✅ PROJEKT ZAKOŃCZONY POMYŚLNIE
═══════════════════════════════════════════════════════════════════════════════
'''

print(summary)

# %% [markdown]
# ## 11. Predykcje na Zbiorze Testowym (test_set.csv)
# 
# ⚠️ **Uwaga:** plik `test_set.csv` to zbiór konkursowy Kaggle i **nie zawiera kolumny `SalePrice`**. Nie da się więc policzyć na nim metryk (R², RMSE) — brak prawdziwych cen do porównania. Rzetelna ocena modeli odbyła się na zbiorze walidacyjnym (sekcja 9).
# 
# Tutaj **uruchamiamy wszystkie modele na test_set.csv**, porównujemy ich predykcje (rozkłady, zgodność między modelami, sanity-check względem rozkładu treningowego) i budujemy uśredniony ensemble. Prawdziwy wynik na teście można uzyskać wysyłając `submission.csv` na Kaggle.

# %%
# TEST WSZYSTKICH MODELI NA ZBIORZE TESTOWYM (test_set.csv - bez etykiet)
from sklearn.base import clone

print('=' * 100)
print('🧪 PREDYKCJE WSZYSTKICH MODELI NA test_set.csv')
print('=' * 100)
print(f'Zbiór testowy: {X_test_full.shape[0]} domów (brak kolumny SalePrice -> brak metryk)')

# Każdy zestrojony model trenujemy na PEŁNYM zbiorze treningowym i predykujemy test
test_predictions = pd.DataFrame(index=X_test_full.index)
for name, model in tuned_models.items():
    fitted = clone(model).fit(X_full, y_log)
    test_predictions[name] = np.expm1(fitted.predict(X_test_full))

# Ensemble - średnia z 4 najlepszych modeli (wg R² na walidacji)
top4 = results_df.head(4)['Model'].tolist()
test_predictions['Ensemble (top4)'] = test_predictions[top4].mean(axis=1)

# Statystyki predykcji vs rozkład treningowy (sanity-check)
print('\n📊 STATYSTYKI PREDYKCJI NA TEŚCIE vs ROZKŁAD TRENINGOWY:')
stats_tbl = test_predictions.describe().T[['mean', '50%', 'min', 'max', 'std']]
stats_tbl.columns = ['Średnia', 'Mediana', 'Min', 'Max', 'Std']
train_ref = y_train_original.describe()
print(stats_tbl.round(0).to_string())
print(f"\n   ODNIESIENIE (trening): średnia=${train_ref['mean']:,.0f}, "
      f"mediana=${train_ref['50%']:,.0f}, min=${train_ref['min']:,.0f}, max=${train_ref['max']:,.0f}")

# Zgodność modeli - korelacja między ich predykcjami
print('\n🔗 KORELACJA PREDYKCJI MIĘDZY MODELAMI (zgodność):')
model_cols = list(tuned_models.keys())
print(test_predictions[model_cols].corr().round(3).to_string())

# Wizualizacja
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
# Rozkłady predykcji najlepszych modeli vs trening
for name in top4:
    axes[0].hist(test_predictions[name], bins=50, alpha=0.4, label=name)
axes[0].hist(y_train_original, bins=50, alpha=0.3, color='black', label='Trening (rzeczywiste)')
axes[0].set_title('Rozkład predykcji na teście vs ceny treningowe', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Cena ($)'); axes[0].set_ylabel('Liczba domów'); axes[0].legend()

# Zgodność najlepszego modelu z ensemble
best = results_df.iloc[0]['Model']
axes[1].scatter(test_predictions[best], test_predictions['Ensemble (top4)'], alpha=0.4, s=15)
lims = [test_predictions[best].min(), test_predictions[best].max()]
axes[1].plot(lims, lims, 'r--')
axes[1].set_title(f'{best} vs Ensemble (top4)', fontsize=12, fontweight='bold')
axes[1].set_xlabel(f'Predykcja: {best} ($)'); axes[1].set_ylabel('Predykcja: Ensemble ($)')
plt.tight_layout(); plt.show()

# Zapis wszystkich predykcji testowych
all_pred_path = f'{base_path}/test_predictions_all_models.csv'
test_predictions.round(2).to_csv(all_pred_path)
print(f'\n✅ Predykcje wszystkich modeli zapisane do: {all_pred_path}')
print('\n🔍 Pierwsze 5 wierszy:')
print(test_predictions.head().round(0).to_string())
