from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

POSITIVE_COLOR, NEGATIVE_COLOR = "#2a78d6", "#eb6834"
CENTER_LAT, CENTER_LON = 42.6977, 23.3219


def to_test_1(X_train, X_test, y_train, y_test):
    # Евклидовото разстояние дава еднаква тежест на всички 20 дни. Съседите
    # приличат на прозореца като цяло, но последният им ден се различава от
    # днешния средно с около 56 GWh, а утрешното потребление зависи най-много от
    # днешното. Затова 5-NN с целите прозорци греши повече от линейната регресия.
    # След изваждането на последния ден се сравнява само формата: кои дни от
    # седмицата са по-високи и кои по-ниски. Прогнозата започва от днешното
    # потребление и добавя средната промяна при съседите. Така 5-NN става малко
    # по-точен от линейната регресия.
    knn = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
    print(f"5-NN с целите прозорци: средна грешка {mean_absolute_error(y_test, knn.predict(X_test)):.0f} GWh")

    shape_knn = KNeighborsRegressor(n_neighbors=5).fit(X_train - X_train[:, -1:], y_train - X_train[:, -1])
    shape_predicted = X_test[:, -1] + shape_knn.predict(X_test - X_test[:, -1:])
    print(f"5-NN с формата на прозорците: средна грешка {mean_absolute_error(y_test, shape_predicted):.0f} GWh")

    linear = LinearRegression().fit(X_train, y_train)
    print(f"Линейна регресия: средна грешка {mean_absolute_error(y_test, linear.predict(X_test)):.0f} GWh")


def to_test_2(homes_train, homes_test):
    # pd.get_dummies прави по една колона с 0 и 1 за всеки квартал.
    # reindex подрежда тестовите колони като тренировъчните и добавя липсващите с 0.
    # One-hot е малко по-точен, защото всеки квартал получава собствена цена,
    # например Бояна и Лозенец са по-скъпи, отколкото предполага разстоянието.
    # Квартал, който го няма в обучението, получава само нули: моделът не знае
    # нищо за него. Разстоянието работи за всеки адрес, за който има координати.
    train_onehot = pd.get_dummies(homes_train["neighborhood"]).astype(int)
    test_onehot = pd.get_dummies(homes_test["neighborhood"]).reindex(columns=train_onehot.columns, fill_value=0)
    X_train = pd.concat([homes_train[["area_m2"]], train_onehot], axis=1)
    X_test = pd.concat([homes_test[["area_m2"]], test_onehot.astype(int)], axis=1)
    onehot_model = LinearRegression().fit(X_train, homes_train["price_eur"])
    onehot_error = mean_absolute_error(homes_test["price_eur"], onehot_model.predict(X_test))
    distance_model = LinearRegression().fit(homes_train[["area_m2", "distance_km"]], homes_train["price_eur"])
    distance_error = mean_absolute_error(homes_test["price_eur"],
                                         distance_model.predict(homes_test[["area_m2", "distance_km"]]))
    print(f"Площ + {train_onehot.shape[1]} one-hot колони: средна грешка {onehot_error:.0f} €")
    print(f"Площ + разстояние: средна грешка {distance_error:.0f} €")

    # Нова обява: 70 m² в Обеля (приблизително 42.744, 23.258), квартал извън данните.
    new_onehot = pd.DataFrame([[70] + [0] * train_onehot.shape[1]], columns=X_train.columns)
    north_km = (42.744 - CENTER_LAT) * 111
    east_km = (23.258 - CENTER_LON) * 111 * np.cos(np.radians(CENTER_LAT))
    new_distance = pd.DataFrame({"area_m2": [70], "distance_km": [np.hypot(north_km, east_km)]})
    print(f"70 m² в Обеля ({new_distance['distance_km'][0]:.1f} km): one-hot {onehot_model.predict(new_onehot)[0]:.0f} €,"
          f" с разстояние {distance_model.predict(new_distance)[0]:.0f} €")


def to_test_3(homes_train, homes_test):
    # Всяка екстра става колона с 1, ако е в списъка на обявата, и 0, ако не е.
    # Шестте екстри се вземат от тренировъчните обяви.
    # Гаражът, паркомястото и терасата са с положителен коефициент, а мазето и
    # таванът с отрицателен. Мазето не прави апартамента по-евтин: мазета и тавани
    # има по-често в старите сгради, а гаражи и паркоместа в новите, които са
    # по-скъпи. Годината липсва в 209 обяви, а екстрите отчасти я заместват.
    extras = sorted(homes_train["extras"].explode().dropna().unique())
    columns = ["area_m2", "distance_km", "condition_rank"]

    def features(frame):
        result = frame[columns].copy()
        for extra in extras:
            result[extra] = frame["extras"].apply(lambda items: int(extra in items))
        return result

    base_model = LinearRegression().fit(homes_train[columns], homes_train["price_eur"])
    base_error = mean_absolute_error(homes_test["price_eur"], base_model.predict(homes_test[columns]))
    model = LinearRegression().fit(features(homes_train), homes_train["price_eur"])
    error = mean_absolute_error(homes_test["price_eur"], model.predict(features(homes_test)))
    print(f"Площ, разстояние и състояние: средна грешка {base_error:.0f} €")
    print(f"Плюс {len(extras)} колони за екстрите: средна грешка {error:.0f} €")
    has_extra = {extra: homes_train["extras"].apply(lambda items: extra in items) for extra in extras}
    table = pd.DataFrame({
        "коефициент (€)": model.coef_[len(columns):].round(0),
        "обяви": [int(has_extra[extra].sum()) for extra in extras],
        "средна година на сградата": [round(homes_train.loc[has_extra[extra], "building.year_built"].mean())
                                      for extra in extras],
    }, index=extras)
    print(table.sort_values("коефициент (€)"))


def to_test_4(rides_train, rides_test):
    # Всяка комбинация от час и вид ден получава собствено тегло, затова моделът
    # научава отделна крива за работните и за почивните дни. Средната грешка пада
    # от около 108 на около 92 велосипеда.
    def features(frame):
        result = pd.DataFrame({"year": frame["year"] - 2011})
        for hour in range(24):
            result[f"work_{hour}"] = ((frame["hour"] == hour) & (frame["workingday"] == 1)).astype(int)
            result[f"rest_{hour}"] = ((frame["hour"] == hour) & (frame["workingday"] == 0)).astype(int)
        return result

    model = LinearRegression().fit(features(rides_train), rides_train["count"])
    error = mean_absolute_error(rides_test["count"], model.predict(features(rides_test)))
    print(f"Час и вид ден, 48 колони: средна грешка {error:.0f} велосипеда")


def to_test_5(X_train, y_train, vocabulary):
    # Булевата маска избира редовете на един клас, а sum(axis=0) събира колоните:
    # колко пъти общо се среща всяка дума. Думите са подредени като колоните.
    # "call", "your", "free", "2", "or" и "now" са сред най-честите в спама, но ги
    # няма сред най-честите в нормалните съобщения, затова помагат. "to" и "a" са
    # чести и в двата класа, а "i", "me" и "my" са типични за разговорите.
    words = np.array(sorted(vocabulary, key=vocabulary.get))
    for label in ["spam", "legit"]:
        totals = X_train[(y_train == label).to_numpy()].sum(axis=0)
        top = np.argsort(totals)[::-1][:10]
        print(f"{label}:", [(str(words[i]), int(totals[i])) for i in top])


def to_test_6(bigram_model, bigram_vocabulary):
    # Биграмите са думите в речника с интервал между двете части. coef_[0] са
    # теглата за класа spam (classes_[1]): положителните теглят към спам.
    # Към спам: "have won", "1 50" (цена £1.50), "stop to" (от "send STOP to"),
    # "to 87077" (кратък номер), "sms ac" (от адреса SMS.ac). Към legit: изрази
    # от разговор като "have you", "like you", "your call", "have to", "i ll" и
    # "lt gt" (HTML кодовете &lt; и &gt;). "call" сама тегли към спам, а
    # "your call" (от "missed your call") към legit: съседната дума променя
    # смисъла. Част от биграмите, например "you is" и "day no", са само в две
    # съобщения и са случайни: перцептронът ги е променил при няколко грешки.
    words = np.array(sorted(bigram_vocabulary, key=bigram_vocabulary.get))
    is_bigram = np.array([" " in word for word in words])
    pairs, pair_weights = words[is_bigram], bigram_model.coef_[0][is_bigram]
    order = np.argsort(pair_weights)
    print("Към спам:", [(str(pairs[i]), float(pair_weights[i])) for i in order[::-1][:10]])
    print("Към legit:", [(str(pairs[i]), float(pair_weights[i])) for i in order[:10]])


# ---------------------------------------------------------------- финална задача

def tokenize(text):
    """Малки букви; всеки символ, който не е буква или цифра, става интервал."""
    cleaned = "".join(symbol.lower() if symbol.isalnum() else " " for symbol in text)
    return cleaned.split()


def document_frequency(token_lists):
    """В колко текста се среща всяка дума."""
    frequency = {}
    for tokens in token_lists:
        for word in set(tokens):
            frequency[word] = frequency.get(word, 0) + 1
    return frequency


def build_vocabulary(token_lists, min_docs=1, stop_words=()):
    """Речник дума: номер на колона, без редките думи и без стоп думите."""
    frequency = document_frequency(token_lists)
    words = sorted(word for word, count in frequency.items()
                   if count >= min_docs and word not in stop_words)
    return {word: column for column, word in enumerate(words)}


def vectorize(token_lists, vocabulary):
    """Bag-of-Words: колко пъти всяка дума от речника се среща във всеки текст."""
    X = np.zeros((len(token_lists), len(vocabulary)), dtype=np.float32)
    for row, tokens in enumerate(token_lists):
        for word in tokens:
            if word in vocabulary:
                X[row, vocabulary[word]] += 1
    return X


def with_bigrams(tokens):
    """Думите плюс всички двойки съседни думи."""
    return tokens + [f"{first} {second}" for first, second in zip(tokens, tokens[1:])]


def train(random_state=42, data_path=None):
    """Отзиви: униграми, униграми без "not" и униграми с биграми."""
    if data_path is None:
        data_path = Path(__file__).resolve().parent / "data" / "reviews.csv"
    reviews = pd.read_csv(data_path).drop_duplicates("text").reset_index(drop=True)
    train_texts, test_texts, y_train, y_test = train_test_split(
        reviews["text"], reviews["label"], test_size=0.2,
        random_state=random_state, stratify=reviews["label"]
    )
    train_tokens = [tokenize(text) for text in train_texts]
    test_tokens = [tokenize(text) for text in test_texts]
    phrases = ["good", "not good", "bad", "not bad"]
    result = {"reviews": reviews, "y_test": y_test, "phrases": phrases, "total": len(y_test),
              "baseline_correct": int(y_test.value_counts().max())}
    variants = {
        "униграми": (lambda tokens: tokens, ()),
        "униграми, not като стоп дума": (lambda tokens: tokens, {"not"}),
        "униграми и биграми": (with_bigrams, ()),
    }
    for name, (prepare, stop_words) in variants.items():
        vocabulary = build_vocabulary([prepare(t) for t in train_tokens], min_docs=2, stop_words=stop_words)
        X_train = vectorize([prepare(t) for t in train_tokens], vocabulary)
        model = Perceptron(random_state=random_state).fit(X_train, y_train)
        # Класовете са по азбучен ред: coef_[0] и decision_function > 0 са за positive.
        predictions = model.predict(vectorize([prepare(t) for t in test_tokens], vocabulary))
        phrase_scores = model.decision_function(vectorize([prepare(tokenize(p)) for p in phrases], vocabulary))
        result[name] = {
            "vocabulary": vocabulary,
            "weights": model.coef_[0],
            "bias": model.intercept_[0],
            "predictions": predictions,
            "correct": int((predictions == y_test).sum()),
            "phrase_scores": phrase_scores,
        }
    return result


def plot_result(result):
    """Думите с най-голямо тегло и оценките на четирите израза."""
    part = result["униграми"]
    words = np.array(sorted(part["vocabulary"], key=part["vocabulary"].get))
    order = np.argsort(part["weights"])
    chosen = np.concatenate([order[:10], order[-10:]])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained",
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    ax1.barh(words[chosen], part["weights"][chosen], color=[NEGATIVE_COLOR] * 10 + [POSITIVE_COLOR] * 10)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set(xlabel="Тегло (вляво към negative, вдясно към positive)", title="Думи с най-голямо тегло")

    names = ["униграми", "униграми и биграми"]
    positions = np.arange(len(result["phrases"]))
    # Други цветове, защото синьото и оранжевото вляво означават класовете.
    for offset, name, color in zip([-0.2, 0.2], names, ["#b0b6bd", "#168c81"]):
        ax2.barh(positions + offset, result[name]["phrase_scores"], height=0.4, color=color, label=name)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set(yticks=positions, yticklabels=result["phrases"], xlabel="Оценка (> 0 означава positive)",
            title="Четирите израза")
    ax2.invert_yaxis()
    ax2.legend()
    return fig


def main():
    result = train()
    reviews = result["reviews"]
    print("Изречения без повторения:", len(reviews))
    print(pd.crosstab(reviews["source"], reviews["label"]))
    print(f"\nБазов модел: {result['baseline_correct']} от {result['total']}")
    for name in ["униграми", "униграми, not като стоп дума", "униграми и биграми"]:
        part = result[name]
        scores = ", ".join(f"{p}: {s:.0f}" for p, s in zip(result["phrases"], part["phrase_scores"]))
        print(f"{name}: речник {len(part['vocabulary'])}, правилни {part['correct']} от {result['total']} | {scores}")
    part = result["униграми"]
    words = np.array(sorted(part["vocabulary"], key=part["vocabulary"].get))
    order = np.argsort(part["weights"])
    print("\nНай-положителни думи:", words[order[::-1][:10]].tolist())
    print("Най-отрицателни думи:", words[order[:10]].tolist())
    bigrams = result["униграми и биграми"]
    for word in ["not", "good", "bad", "not good", "not bad"]:
        column = bigrams["vocabulary"].get(word)
        weight = "няма го в речника" if column is None else bigrams["weights"][column]
        print(f"Тегло на {word} (с биграми): {weight}")
    plot_result(result)
    plt.show()


if __name__ == "__main__":
    main()
