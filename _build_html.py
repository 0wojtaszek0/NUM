# Buduje samodzielną prezentację HTML (CSS+JS, obrazy w base64) - prosty język polski
import json, base64, os

base = os.path.dirname(__file__)
A = f'{base}/prezentacja_assets'
R = json.load(open(f'{A}/results.json'))


def img(name):
    with open(f'{A}/{name}', 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()

IMGS = {n: img(f'{n}.png') for n in
        ['01_dist', '02_corr', '03_outliers', '04_ranking', '05_residuals', '06_shap', '07_test',
         '08_heatmap', '09_histograms', '10_boxplots']}

# Linki do Wikipedii (zweryfikowane)
W = {
    'PCA': 'https://pl.wikipedia.org/wiki/Analiza_g%C5%82%C3%B3wnych_sk%C5%82adowych',
    'R2': 'https://pl.wikipedia.org/wiki/Wsp%C3%B3%C5%82czynnik_determinacji',
    'RMSE': 'https://pl.wikipedia.org/wiki/B%C5%82%C4%85d_%C5%9Bredniokwadratowy',
    'CV': 'https://pl.wikipedia.org/wiki/Sprawdzian_krzy%C5%BCowy',
    'RF': 'https://pl.wikipedia.org/wiki/Las_losowy',
    'SKOS': 'https://pl.wikipedia.org/wiki/Sko%C5%9Bno%C5%9B%C4%87',
    'MEDIANA': 'https://pl.wikipedia.org/wiki/Mediana',
    'REGR': 'https://pl.wikipedia.org/wiki/Regresja_liniowa',
    'SHAP': 'https://pl.wikipedia.org/wiki/Warto%C5%9B%C4%87_Shapleya',
    'GB': 'https://en.wikipedia.org/wiki/Gradient_boosting',
    'XGB': 'https://en.wikipedia.org/wiki/XGBoost',
    'ISO': 'https://en.wikipedia.org/wiki/Isolation_forest',
    'VIF': 'https://en.wikipedia.org/wiki/Variance_inflation_factor',
    'LEAK': 'https://en.wikipedia.org/wiki/Leakage_(machine_learning)',
    'XAI': 'https://pl.wikipedia.org/wiki/Wyt%C5%82umaczalna_sztuczna_inteligencja',
}


def t(key, label):
    """Pojęcie z odnośnikiem do encyklopedii."""
    return (f'<a class="term" href="{W[key]}" target="_blank" rel="noopener" '
            f'title="Otwórz wyjaśnienie pojęcia (Wikipedia)">{label}</a>')

rank = R['ranking']
best, second = rank[0], rank[1]
n_all_feat = R['n_num'] + R['n_cat']

# tabela rankingu
rows = ''.join(
    f'<tr class="{"win" if i==0 else ""}"><td>{i+1}</td><td>{r["Model"]}</td>'
    f'<td>{r["R2"]:.3f}</td><td>${r["RMSE"]:,.0f}</td><td>${r["MAE"]:,.0f}</td>'
    f'<td>{r["MAPE"]*100:.1f}%</td></tr>'
    for i, r in enumerate(rank))

# ---------- SLAJDY ----------
slides = []

slides.append(f'''
<section class="slide title">
  <div class="kicker">Projekt zaliczeniowy · uczenie maszynowe</div>
  <h1>Analiza i przewidywanie cen domów</h1>
  <p class="lead">Budujemy i porównujemy modele, które na podstawie cech nieruchomości
     szacują jej cenę sprzedaży. Dane: zbiór „Ames Housing".</p>
  <div class="chips">
    <span>{R['n_train_raw']} domów do nauki</span>
    <span>{n_all_feat} cech opisujących dom</span>
    <span>8 porównywanych modeli</span>
  </div>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Wprowadzenie</span>
  <h2>O czym jest ten projekt</h2>
  <ul>
    <li><b>Cel:</b> nauczyć komputer szacować cenę domu na podstawie jego opisu
        (powierzchnia, jakość, garaż, rok budowy itd.).</li>
    <li><b>Dane do nauki:</b> {R['n_train_raw']} domów wraz ze znaną ceną sprzedaży.</li>
    <li><b>Dane do sprawdzenia:</b> {R['n_test']} domów <u>bez podanej ceny</u>
        (cena jest ukryta — to zbiór konkursowy).</li>
    <li><b>Cechy:</b> {R['n_num']} liczbowych i {R['n_cat']} opisowych (np. dzielnica, rodzaj garażu).</li>
    <li><b>Główna trudność:</b> ceny są bardzo nierówno rozłożone — kilka domów jest
        skrajnie drogich ({t('SKOS','skośność rozkładu')} = {R['skew_raw']:.2f}).</li>
  </ul>
  <div class="note">Skoro zbiór testowy nie ma cen, rzetelnej oceny modeli dokonujemy na
     osobno wydzielonej części danych do nauki.</div>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Co opisują dane</span>
  <h2>Najważniejsze zmienne (słownik danych)</h2>
  <div class="dict">
    <div class="grp"><h3>Powierzchnia</h3>
      <p><b>GrLivArea</b> — powierzchnia mieszkalna nad ziemią</p>
      <p><b>TotalBsmtSF</b> — powierzchnia piwnicy</p>
      <p><b>1stFlrSF / 2ndFlrSF</b> — powierzchnia parteru / piętra</p>
      <p><b>LotArea</b> — powierzchnia działki</p>
    </div>
    <div class="grp"><h3>Jakość i stan</h3>
      <p><b>OverallQual</b> — ogólna jakość wykończenia (skala 1–10)</p>
      <p><b>OverallCond</b> — ogólny stan domu (1–10)</p>
      <p><b>KitchenQual</b> — jakość kuchni</p>
      <p><b>ExterQual</b> — jakość elewacji</p>
    </div>
    <div class="grp"><h3>Pomieszczenia i garaż</h3>
      <p><b>FullBath / HalfBath</b> — łazienki pełne / połówkowe</p>
      <p><b>BedroomAbvGr</b> — liczba sypialni</p>
      <p><b>TotRmsAbvGrd</b> — łączna liczba pokoi</p>
      <p><b>GarageCars / GarageArea</b> — miejsca / powierzchnia garażu</p>
    </div>
    <div class="grp"><h3>Czas i lokalizacja</h3>
      <p><b>YearBuilt</b> — rok budowy</p>
      <p><b>YearRemodAdd</b> — rok ostatniego remontu</p>
      <p><b>Neighborhood</b> — dzielnica miasta</p>
      <p><b>MSZoning</b> — rodzaj strefy zabudowy</p>
    </div>
  </div>
  <div class="note"><b>SalePrice</b> — cena sprzedaży domu; to właśnie ją model ma przewidywać.</div>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Co opisują dane</span>
  <h2>Nowe cechy stworzone w projekcie</h2>
  <p class="lead" style="margin-bottom:18px">Z istniejących kolumn policzyliśmy dodatkowe, sensowne wielkości
     oddające wiedzę „rynkową".</p>
  <ul class="two">
    <li><b>TotalSF</b> — łączna powierzchnia (piwnica + część mieszkalna).</li>
    <li><b>TotalBath</b> — łączna liczba łazienek (połówki liczone jako 0,5).</li>
    <li><b>HouseAge</b> — wiek domu w chwili sprzedaży (rok sprzedaży − rok budowy).</li>
    <li><b>RemodAge</b> — ile lat od ostatniego remontu.</li>
    <li><b>QualityCondition</b> — jakość × stan (połączenie dwóch ocen).</li>
    <li><b>GarageAreaPerCar</b> — powierzchnia garażu na jedno auto.</li>
    <li><b>RoomSize</b> — średnia wielkość pokoju.</li>
    <li><b>TotalOutdoorSF</b> — łączna powierzchnia tarasów i ganków.</li>
  </ul>
  <div class="note"><b>Ważne zastrzeżenie:</b> część tych cech (np. <b>TotalSF</b>, QualityCondition)
     mocno powiela informację z kolumn źródłowych. Dlatego krok kontroli współliniowości
     ({t('VIF','wskaźnik VIF')}) celowo je usuwa, a w modelach liniowych dodatkowo redukuje je {t('PCA','PCA')}.
     Do modeli trafiają więc głównie cechy wnoszące nową informację (np. wiek domu, lata od remontu).</div>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Plan pracy</span>
  <h2>Jak wygląda cały proces</h2>
  <ul class="steps">
    <li><span>1</span> Poznanie danych — rozkłady, zależności, brakujące wartości.</li>
    <li><span>2</span> Uzupełnianie braków z myślą o znaczeniu (brak basenu ≠ pomyłka).</li>
    <li><span>3</span> Tworzenie nowych, sensownych cech (np. łączna powierzchnia, wiek domu).</li>
    <li><span>4</span> Wykrywanie i usuwanie wartości skrajnych ({t('ISO','metoda Isolation Forest')}).</li>
    <li><span>5</span> Wybór cech i przygotowanie danych w jednym, spójnym potoku.</li>
    <li><span>6</span> Uczenie 8 modeli i dobieranie ich ustawień.</li>
    <li><span>7</span> Ocena na danych, których model nie widział + analiza błędów.</li>
    <li><span>8</span> Wyjaśnienie decyzji modelu i przewidywania dla nowych domów.</li>
  </ul>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Poznanie danych</span>
  <h2>Jak rozłożone są ceny domów</h2>
  <img src="{IMGS['01_dist']}" alt="Rozkład cen">
  <div class="cap">Ceny są „rozciągnięte" w prawo. Po przekształceniu logarytmicznym rozkład
     staje się symetryczny — dzięki temu modele uczą się dużo lepiej i nie są nadmiernie
     karane za błędy przy najdroższych domach.</div>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Poznanie danych</span>
  <h2>Co najmocniej wpływa na cenę</h2>
  <img src="{IMGS['02_corr']}" alt="Korelacje cech z ceną">
  <div class="cap">Najsilniej z ceną wiążą się: ogólna jakość wykończenia, łączna powierzchnia
     (TotalSF) i powierzchnia mieszkalna oraz wielkość garażu.</div>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Poznanie danych</span>
  <h2>Mapa ciepła — zależności między cechami</h2>
  <img src="{IMGS['08_heatmap']}" alt="Mapa ciepła korelacji">
  <div class="cap">Im ciemniejszy niebieski, tym silniejsza zależność dodatnia. Widać też,
     że niektóre cechy są ze sobą mocno powiązane (np. powierzchnia i garaż) — to nadmiarowość,
     którą kontroluje {t('VIF','wskaźnik VIF')}. Wiek domu (HouseAge) działa odwrotnie — starszy dom, niższa cena.</div>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Poznanie danych</span>
  <h2>Histogramy — jak rozłożone są cechy</h2>
  <img src="{IMGS['09_histograms']}" alt="Histogramy cech">
  <div class="cap">Większość cech powierzchniowych jest „rozciągnięta" w prawo (kilka bardzo dużych domów),
     a rok budowy pokazuje wyraźny wzrost liczby nowszych nieruchomości.</div>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Poznanie danych</span>
  <h2>Wykresy pudełkowe — cena a kluczowe cechy</h2>
  <img src="{IMGS['10_boxplots']}" alt="Wykresy pudełkowe">
  <div class="cap">Pudełko pokazuje typowy zakres cen, a linia w środku to {t('MEDIANA','mediana')}.
     Im wyższa jakość domu i większy garaż, tym wyraźnie wyższa cena — zależność jest bardzo silna.</div>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Przygotowanie danych</span>
  <h2>Mądre uzupełnianie braków</h2>
  <ul>
    <li><b>Problem:</b> wstawianie „przeciętnej" wartości w miejsce braku potrafi zafałszować
        rzeczywistość.</li>
    <li class="sub">Np. dom bez garażu dostałby przeciętną powierzchnię garażu i sztucznie „zyskałby" na wartości.</li>
    <li><b>Nasze podejście — rozróżniamy rodzaj braku:</b></li>
    <li class="sub">Brak <u>oznaczający nieobecność</u> (brak basenu, garażu, piwnicy):
        wpisujemy „brak" lub wartość 0.</li>
    <li class="sub">Brak <u>przypadkowy</u> (np. niewpisana szerokość działki): uzupełniamy
        {t('MEDIANA','medianą')} — i to dopiero wewnątrz potoku uczenia.</li>
    <li><b>Efekt:</b> liczba braków spadła z {R['miss_before']:,} do {R['miss_after']}.</li>
  </ul>
  <div class="note">Stałe wartości („brak"/0) są bezpieczne — nie powodują {t('LEAK','wycieku informacji')} ze zbioru testowego.</div>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Przygotowanie danych</span>
  <h2>Usuwanie wartości skrajnych</h2>
  <img src="{IMGS['03_outliers']}" alt="Wykrywanie wartości odstających">
  <div class="cap">Przyjęliśmy ostrożny próg 1% najbardziej nietypowych obserwacji — to {R['n_outliers']} domów
     o nierynkowym stosunku powierzchni do ceny (m.in. znane z tego zbioru ogromne, a tanie domy
     o powierzchni &gt;4000 stóp²). Usuwamy je tylko z danych do nauki ({R['n_train_raw']} → {R['n_train_clean']});
     zbioru testowego nie ruszamy. To celowo zachowawcze cięcie, nie „podkręcanie" wyniku.</div>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Przygotowanie danych</span>
  <h2>Dwa sposoby przygotowania danych</h2>
  <ul>
    <li>Cenę uczymy w skali logarytmicznej — sprawiedliwsza ocena błędów dla drogich domów.</li>
    <li>Całe przygotowanie danych zamknęliśmy w jednym potoku, dobranym do rodzaju modelu:</li>
    <li class="sub"><b>Ścieżka A</b> (modele oparte na {t('REGR','regresji')} i sąsiadach):
        ujednolicenie skali + {t('PCA','analiza głównych składowych (PCA)')} → {R['pca_dims']} nowych zmiennych
        zachowujących 95% informacji.</li>
    <li class="sub"><b>Ścieżka B</b> (modele drzewiaste): bez PCA → pełne {R['n_feat_tree']} cech.</li>
    <li>PCA pomaga modelom liniowym, ale szkodzi drzewom i utrudnia wyjaśnianie —
        dlatego stosujemy je tylko tam, gdzie naprawdę pomaga.</li>
  </ul>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Uczenie modeli</span>
  <h2>Uczciwa ocena modeli</h2>
  <ul>
    <li>Porównaliśmy 8 modeli — od prostej {t('REGR','regresji')} po {t('GB','wzmacnianie gradientowe')}
        i {t('XGB','XGBoost')}.</li>
    <li>Dla każdego modelu dobraliśmy najlepsze ustawienia, korzystając z
        {t('CV','sprawdzianu krzyżowego')} (pięciokrotnego).</li>
    <li><b>Najważniejsze:</b> oceniamy modele na osobnej części danych, której
        <u>nie widziały podczas nauki</u> — nigdy na danych treningowych.</li>
    <li class="sub">Przygotowanie danych liczone jest od nowa w każdej części sprawdzianu —
        to eliminuje {t('LEAK','wyciek informacji')}.</li>
    <li>Wybór cech wspomógł też {t('VIF','wskaźnik VIF')}, który wykrywa cechy nadmiernie powtarzające informację.</li>
  </ul>
  <div class="note">Wcześniejsza wersja oceniała modele na danych użytych do nauki — to sztucznie zawyżało wyniki.</div>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Wyniki</span>
  <h2>Który model wypadł najlepiej</h2>
  <img src="{IMGS['04_ranking']}" alt="Ranking modeli">
  <div class="cap">Najlepszy: <b>{best['Model']}</b> — {t('R2','R²')} = {best['R2']:.3f},
     {t('RMSE','RMSE')} = ${best['RMSE']:,.0f}, średni błąd ≈ ${best['MAE']:,.0f}.
     Modele drzewiaste wygrywają, bo wychwytują <b>nieliniowe</b> zależności (np. cena rośnie z jakością
     coraz szybciej) i radzą sobie ze współzależnymi cechami — modele liniowe tego nie potrafią.</div>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Wyniki</span>
  <h2>Pełne zestawienie modeli</h2>
  <table class="rank">
    <thead><tr><th>#</th><th>Model</th><th>{t('R2','R²')}</th><th>{t('RMSE','RMSE')}</th><th>Średni błąd</th><th>Błąd %</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <div class="cap">{t('R2','R²')} mówi, jaką część zmienności cen model wyjaśnia (im bliżej 1, tym lepiej).
     RMSE, średni błąd oraz „Błąd %" (MAPE) policzono na <b>rzeczywistych cenach w dolarach</b>
     — po odwróceniu logarytmu. Dlatego „Błąd %" = 11% oznacza naprawdę 11% pomyłki w cenie.</div>
</section>''')

bias = 'zaniża' if R['resid_bias'] > 0 else 'zawyża'
slides.append(f'''
<section class="slide">
  <span class="tag">Sprawdzenie modelu</span>
  <h2>Gdzie model się myli</h2>
  <img src="{IMGS['05_residuals']}" alt="Analiza błędów">
  <div class="cap">Po lewej: przewidywania kontra rzeczywiste ceny (im bliżej linii, tym lepiej).
     Po prawej: rozkład błędów — skupiony wokół zera. Średni błąd ≈ ${R['mae_val']:,.0f},
     typowy błąd to ok. {R['mape_med']:.1f}%; model lekko {bias} ceny.</div>
</section>''')

tops = ', '.join(x[0] for x in R['shap_top'][:4])
slides.append(f'''
<section class="slide">
  <span class="tag">Wyjaśnialność</span>
  <h2>Dlaczego model podejmuje takie decyzje</h2>
  <img src="{IMGS['06_shap']}" alt="Wykres SHAP">
  <div class="cap">Wykres oparty na {t('SHAP','wartościach Shapleya')} pokazuje, które cechy najmocniej
     podnoszą lub obniżają wycenę. Najważniejsze okazały się: {tops}.
     Dzięki temu model nie jest „czarną skrzynką".</div>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Wyniki</span>
  <h2>Przewidywania dla nowych domów</h2>
  <img src="{IMGS['07_test']}" alt="Przewidywania na zbiorze testowym">
  <div class="cap">Zbiór testowy nie ma prawdziwych cen, więc nie liczymy na nim ocen.
     Rozkład przewidywań (średnia ≈ ${R['test_mean']:,.0f}) pokrywa się jednak z cenami
     z danych do nauki — to dobry znak, że model działa rozsądnie.</div>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Co poprawiliśmy</span>
  <h2>Najważniejsze ulepszenia jakości</h2>
  <ul class="two">
    <li>✔ Brak {t('LEAK','wycieku informacji')} — całe przygotowanie danych w jednym potoku.</li>
    <li>✔ Uczciwa ocena — na danych nieużytych do nauki + {t('CV','sprawdzian krzyżowy')}.</li>
    <li>✔ Mądre uzupełnianie braków — „brak"/0 zamiast ślepej średniej.</li>
    <li>✔ Usuwanie wartości skrajnych ({t('ISO','Isolation Forest')}).</li>
    <li>✔ Logarytmiczna skala ceny.</li>
    <li>✔ {t('PCA','PCA')} tylko dla modeli liniowych, drzewa na pełnych cechach.</li>
    <li>✔ Dobór ustawień modeli (sprawdzian krzyżowy).</li>
    <li>✔ Analiza błędów i wyjaśnianie decyzji ({t('SHAP','SHAP')}).</li>
  </ul>
</section>''')

slides.append(f'''
<section class="slide">
  <span class="tag">Podsumowanie</span>
  <h2>Wnioski i co dalej</h2>
  <ul>
    <li>Najlepsze są modele drzewiaste: <b>{best['Model']}</b> ({t('R2','R²')} = {best['R2']:.3f})
        i {second['Model']} ({second['R2']:.3f}).</li>
    <li>Wyprzedzają one modele liniowe (R² ok. 0,84 wobec ok. 0,70).</li>
    <li>Modele liniowe potrafią podać nierealne ceny dla nietypowych domów —
        modele drzewiaste są bezpieczniejsze.</li>
    <li>Najwięcej dla wyceny znaczą: jakość wykończenia, łączna powierzchnia i wiek domu.</li>
    <li><b>Rekomendacja:</b> wdrożyć {t('GB','wzmacnianie gradientowe')} / {t('XGB','XGBoost')}
        i obserwować błędy w praktyce.</li>
    <li><b>Dalej:</b> łączenie modeli, lepsze uzupełnianie braków wg dzielnicy, więcej cech opisowych.</li>
  </ul>
</section>''')

slides.append('''
<section class="slide title end">
  <h1>Dziękuję za uwagę</h1>
  <p class="lead">Kompletny, przejrzysty i wyjaśnialny proces przewidywania cen domów.</p>
  <div class="chips"><span>Pytania mile widziane</span></div>
</section>''')

SLIDES_HTML = '\n'.join(slides)
N = len(slides)

HTML = f'''<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Analiza cen domów — prezentacja</title>
<style>
  :root {{
    --ink:#2F3B4C; --bg:#F7F5F1; --blue:#4C72B0; --green:#55A868;
    --red:#C44E52; --sand:#CCB974; --grey:#6B7280; --card:#FFFFFF;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  html,body {{ height:100%; }}
  body {{
    font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    background:linear-gradient(135deg,#33414f,#2a3540); color:var(--ink);
    overflow:hidden;
  }}
  .deck {{ position:fixed; inset:0; }}
  .slide {{
    position:absolute; inset:0; display:none; flex-direction:column;
    justify-content:center; padding:5.2vh 8vw 8vh; background:var(--bg);
    animation:fade .45s ease; border-left:10px solid var(--blue);
  }}
  .slide.active {{ display:flex; }}
  @keyframes fade {{ from{{opacity:0; transform:translateY(12px);}} to{{opacity:1; transform:none;}} }}

  h1 {{ font-size:clamp(28px,4.4vw,52px); line-height:1.08; margin-bottom:18px; }}
  h2 {{ font-size:clamp(24px,3.3vw,40px); line-height:1.12; margin-bottom:22px;
        padding-bottom:12px; border-bottom:3px solid var(--green); display:inline-block; }}
  .tag {{ color:var(--blue); font-weight:700; letter-spacing:2px; text-transform:uppercase;
          font-size:clamp(11px,1.2vw,14px); margin-bottom:8px; }}
  .kicker {{ color:var(--sand); font-weight:700; letter-spacing:3px; text-transform:uppercase;
             font-size:14px; margin-bottom:14px; }}
  .lead {{ font-size:clamp(16px,1.9vw,23px); color:var(--grey); max-width:65ch; line-height:1.5; }}

  ul {{ list-style:none; max-width:74ch; }}
  ul li {{ position:relative; padding-left:30px; margin-bottom:14px;
           font-size:clamp(15px,1.7vw,21px); line-height:1.45; }}
  ul li::before {{ content:"●"; color:var(--blue); position:absolute; left:0; top:1px; font-size:.8em; }}
  ul li.sub {{ padding-left:48px; color:var(--grey); font-size:clamp(13px,1.45vw,18px); margin-bottom:8px; }}
  ul li.sub::before {{ content:"–"; left:24px; color:var(--sand); }}
  ul.steps li {{ padding-left:46px; }}
  ul.steps li span {{ position:absolute; left:0; top:-2px; width:30px; height:30px; border-radius:50%;
        background:var(--blue); color:#fff; display:flex; align-items:center; justify-content:center;
        font-weight:700; font-size:15px; }}
  ul.steps li::before {{ content:none; }}
  ul.two {{ max-width:100%; columns:2; column-gap:50px; }}
  ul.two li {{ break-inside:avoid; }}
  ul.two li::before {{ content:none; }}

  img {{ max-width:100%; max-height:60vh; object-fit:contain; align-self:center;
         border-radius:10px; box-shadow:0 10px 30px rgba(47,59,76,.18); }}
  .cap {{ margin-top:18px; color:var(--green); font-weight:600; font-size:clamp(13px,1.5vw,18px);
          max-width:90ch; line-height:1.4; }}
  .cap::before {{ content:"➜  "; }}
  .note {{ margin-top:20px; background:#EEEAE2; border-left:5px solid var(--sand);
           padding:12px 18px; border-radius:8px; color:var(--ink); max-width:80ch;
           font-size:clamp(13px,1.45vw,17px); }}
  .chips {{ margin-top:30px; display:flex; gap:14px; flex-wrap:wrap; }}
  .chips span {{ background:#EDEAE3; border:1px solid #Dcd6cc; padding:9px 16px; border-radius:30px;
                 font-weight:600; font-size:clamp(12px,1.4vw,16px); color:var(--ink); }}

  table.rank {{ border-collapse:collapse; width:100%; max-width:980px; font-size:clamp(12px,1.5vw,18px); }}
  table.rank th, table.rank td {{ padding:9px 14px; text-align:left; border-bottom:1px solid #E0DBD1; }}
  table.rank th {{ color:var(--blue); border-bottom:2px solid var(--blue); }}
  table.rank td:first-child, table.rank th:first-child {{ width:40px; color:var(--grey); }}
  table.rank tr.win td {{ background:#EAF1EA; font-weight:700; }}

  .term {{ color:var(--blue); text-decoration:none; border-bottom:2px dotted var(--blue);
           cursor:help; }}
  .term:hover {{ background:#E7EEF6; }}

  /* słownik danych */
  .dict {{ display:grid; grid-template-columns:1fr 1fr; gap:14px 46px; max-width:100%; }}
  .grp h3 {{ color:var(--blue); font-size:clamp(15px,1.7vw,21px); margin-bottom:6px; }}
  .grp p {{ font-size:clamp(12px,1.35vw,16px); margin-bottom:3px; line-height:1.34; color:var(--ink); }}
  .grp p b {{ color:var(--green); }}

  /* tytułowe slajdy - ciemne tło */
  .slide.title {{ background:linear-gradient(135deg,#2F3B4C,#3a4a5e); color:#fff;
                  border-left-color:var(--green); }}
  .slide.title h1 {{ color:#fff; }}
  .slide.title .lead {{ color:#D7DEE7; }}
  .slide.title .chips span {{ background:rgba(255,255,255,.1); border-color:rgba(255,255,255,.2); color:#fff; }}
  .slide.title.end {{ justify-content:center; text-align:center; align-items:center; }}

  /* nawigacja */
  #bar {{ position:fixed; top:0; left:0; height:5px; background:var(--green); width:0;
          transition:width .3s; z-index:50; }}
  #nav {{ position:fixed; bottom:18px; right:22px; display:flex; align-items:center; gap:14px;
          z-index:50; background:rgba(47,59,76,.82); padding:8px 14px; border-radius:30px; }}
  #nav button {{ background:#fff; color:var(--ink); border:none; width:38px; height:38px;
          border-radius:50%; font-size:18px; cursor:pointer; transition:.2s; }}
  #nav button:hover {{ background:var(--sand); }}
  #count {{ color:#fff; font-weight:600; font-size:14px; min-width:54px; text-align:center; }}
  #hint {{ position:fixed; bottom:22px; left:22px; color:rgba(255,255,255,.65); font-size:12px; z-index:50; }}
  @media print {{ body{{overflow:visible;}} .slide{{display:flex; position:relative; page-break-after:always; height:100vh;}} #nav,#bar,#hint{{display:none;}} }}
</style>
</head>
<body>
  <div id="bar"></div>
  <div class="deck">{SLIDES_HTML}</div>
  <div id="hint">← → spacja · F = pełny ekran</div>
  <div id="nav">
    <button id="prev" aria-label="Poprzedni">‹</button>
    <span id="count"></span>
    <button id="next" aria-label="Następny">›</button>
  </div>
<script>
  const slides=[...document.querySelectorAll('.slide')];
  let i=0; const bar=document.getElementById('bar'), count=document.getElementById('count');
  function show(n){{
    i=Math.max(0,Math.min(slides.length-1,n));
    slides.forEach((s,k)=>s.classList.toggle('active',k===i));
    bar.style.width=((i+1)/slides.length*100)+'%';
    count.textContent=(i+1)+' / '+slides.length;
    location.hash=i+1;
  }}
  function next(){{show(i+1);}} function prev(){{show(i-1);}}
  document.getElementById('next').onclick=next;
  document.getElementById('prev').onclick=prev;
  document.addEventListener('keydown',e=>{{
    if(['ArrowRight','ArrowDown',' ','PageDown'].includes(e.key)){{e.preventDefault();next();}}
    else if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){{e.preventDefault();prev();}}
    else if(e.key==='Home') show(0);
    else if(e.key==='End') show(slides.length-1);
    else if(e.key==='f'||e.key==='F'){{ if(!document.fullscreenElement) document.documentElement.requestFullscreen(); else document.exitFullscreen(); }}
  }});
  // dotyk (telefon/tablet)
  let x0=null;
  document.addEventListener('touchstart',e=>x0=e.touches[0].clientX,{{passive:true}});
  document.addEventListener('touchend',e=>{{ if(x0===null)return; const dx=e.changedTouches[0].clientX-x0;
    if(Math.abs(dx)>50){{ dx<0?next():prev(); }} x0=null; }});
  // klik w prawą/lewą połowę
  document.querySelector('.deck').addEventListener('click',e=>{{
    if(e.target.closest('a,button'))return;
    (e.clientX>window.innerWidth/2)?next():prev();
  }});
  const start=parseInt(location.hash.replace('#',''))||1; show(start-1);
</script>
</body>
</html>'''

out = f'{base}/Prezentacja_Analiza_Cen_Domow.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(HTML)
print('HTML zapisany:', out)
print('Slajdów:', N, '| rozmiar:', round(len(HTML)/1024), 'KB')