from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#4d4d4d"


def sigmoid(z):
    """Сигмоидната функция 1 / (1 + e⁻ᶻ)."""
    return 1 / (1 + np.exp(-z))


def softmax(z):
    """e^(zₖ) / Σⱼ e^(zⱼ) за всеки клас; работи и за таблица от резултати."""
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


def log_likelihood(b, w, X, y):
    """l(w): сумата от логаритмите на вероятностите на наблюдаваните класове.

    X може да е един признак (w е число) или таблица с няколко признака (w е вектор).
    """
    y = np.asarray(y)
    X = np.asarray(X, dtype=float).reshape(len(y), -1)
    h = sigmoid(b + X @ np.atleast_1d(np.asarray(w, dtype=float)))
    observed = np.where(y == 1, h, 1 - h)
    return float(np.sum(np.log(observed)))


def to_test_1(tenure):
    # partial_fit прави една епоха. По стажа и публикациите преподавателите са
    # линейно разделими: грешките стават 0 на 35-ата епоха и остават 0, защото
    # без грешка перцептронът не променя теглата. Само по стажа те не са
    # разделими: всяка граница греши поне за двама (трима преподаватели имат по
    # 3 години, но не еднакъв клас). Затова теглата продължават да се местят и
    # броят на грешките скача между 2 и 7 без край.
    cases = {"само стаж": ["years"], "стаж и публикации": ["years", "publications"]}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for (name, features), color in zip(cases.items(), [ORANGE, BLUE]):
        model = Perceptron(random_state=42)
        errors = []
        for epoch in range(40):
            model.partial_fit(tenure[features], tenure["tenured"], classes=[0, 1])
            errors.append(int((model.predict(tenure[features]) != tenure["tenured"]).sum()))
        print(f"{name}: грешки по епохи {errors}")
        ax.plot(range(1, 41), errors, marker="o", markersize=4, color=color, label=name)
    ax.set(xlabel="епоха", ylabel="грешно класифицирани", title="Грешки на перцептрона по епохи")
    ax.legend()
    plt.show()


def to_test_2(tenure):
    # Правата минимизира квадратите на остатъците на всички примери. Двамата нови
    # преподаватели са далеч вдясно и правата се накланя към тях: наклонът
    # намалява и мястото, където тя стига 0.5, се мести от 6.1 на 7.2 години,
    # макар че новите примери не казват нищо за хората около границата. За
    # логистичната регресия вероятността при 30 и 35 години и без това е почти 1,
    # новите примери почти не променят правдоподобието и границата остава 5.95.
    data = tenure[["years", "tenured"]]
    extended = pd.concat([data, pd.DataFrame({"years": [30, 35], "tenured": [1, 1]})], ignore_index=True)
    grid = pd.DataFrame({"years": np.linspace(0, 36, 300)})
    rows = []
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(extended["years"], extended["tenured"], s=40, color="black", zorder=3)
    for name, frame, style in [("14 преподаватели", data, ":"), ("с 30 и 35 години", extended, "-")]:
        line = LinearRegression().fit(frame[["years"]], frame["tenured"])
        logistic = LogisticRegression(C=np.inf).fit(frame[["years"]], frame["tenured"])
        b, w = logistic.intercept_[0], logistic.coef_[0, 0]
        rows.append({
            "данни": name,
            "наклон на правата": round(line.coef_[0], 4),
            "правата стига 0.5 при": round((0.5 - line.intercept_) / line.coef_[0], 2),
            "w на логистичната": round(w, 3),
            "-b / w": round(-b / w, 2),
        })
        ax.plot(grid["years"], line.predict(grid), color=ORANGE, linestyle=style, label=f"права, {name}")
        ax.plot(grid["years"], logistic.predict_proba(grid)[:, 1], color=BLUE, linestyle=style,
                label=f"логистична, {name}")
    display(pd.DataFrame(rows))
    ax.axhline(0.5, color="gray", linewidth=0.8)
    ax.set(ylim=(-0.3, 1.5), xlabel="години стаж", ylabel="прогноза",
           title="Далечните примери местят правата, но не и сигмоидата")
    ax.legend(fontsize=8)
    plt.show()


def to_test_3(logit_model):
    # logit(p) = b + w · years, откъдето years = (logit(p) - b) / w.
    # logit(0.9) = ln 9 = 2.197, а logit(0.1) = -2.197. Затова двата отговора са на
    # 2.197 / w = 3.24 години от двете страни на границата -b / w = 5.95.
    b, w = logit_model.intercept_[0], logit_model.coef_[0, 0]
    for p in [0.9, 0.1]:
        years = (np.log(p / (1 - p)) - b) / w
        check = logit_model.predict_proba(pd.DataFrame({"years": [years]}))[0, 1]
        print(f"P = {p}: logit(p) = {np.log(p / (1 - p)):.3f}, years = {years:.2f},"
              f" проверка с predict_proba: {check:.3f}")
    print(f"Граница -b / w = {-b / w:.2f} години; разстояние ln 9 / w = {np.log(9) / w:.2f} години")


def to_test_4(tenure, logit_model):
    # При фиксирано b l(w) е крива с една най-висока точка. Тя е при w на модела:
    # LogisticRegression(C=np.inf) търси точно максимума на правдоподобието.
    # В упражнение 3 RSS имаше най-ниска точка, а тук l(w) има най-висока.
    b, w_model = logit_model.intercept_[0], logit_model.coef_[0, 0]
    ws = np.linspace(0, 2, 300)
    values = np.array([log_likelihood(b, w, tenure["years"], tenure["tenured"]) for w in ws])
    best = ws[values.argmax()]
    print(f"Най-голямо l(w) = {values.max():.3f} при w = {best:.3f}; w на модела = {w_model:.3f}")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(ws, values, color="black")
    ax.axvline(w_model, color=ORANGE, linestyle="--", label="w на LogisticRegression")
    ax.set(xlabel="w (b е от модела)", ylabel="l(w)", title="Логаритъмът на правдоподобието при различни w")
    ax.legend()
    plt.show()


def to_test_5(tenure):
    # По стажа и публикациите всяка права между двата класа ги разделя без
    # грешка, например 5 · стаж + 2 · публикации - 49 = 0. Ако теглата и
    # свободният член се умножат по s, границата остава същата, но кривата става
    # по-стръмна. Всички примери са от правилната страна, затова всеки получава
    # вероятност на своя клас по-близо до 1 и l расте към 0, без да го достигне.
    # Максимум няма. С C=np.inf методът спира, когато подобрението стане
    # незначително: теглата са големи и дори двете изключения получават
    # вероятност почти 1 и почти 0. Регуляризацията (C=1) наказва големите
    # тегла: те остават малки, а вероятностите за изключенията са умерени.
    features = ["years", "publications"]
    exceptions = (((tenure["years"] == 3) & (tenure["tenured"] == 1))
                  | ((tenure["years"] == 7) & (tenure["tenured"] == 0)))
    for C in [np.inf, 1.0]:
        model = LogisticRegression(C=C).fit(tenure[features], tenure["tenured"])
        b = model.intercept_[0]
        w_years, w_pubs = model.coef_[0]
        p_exceptions = model.predict_proba(tenure.loc[exceptions, features])[:, 1]
        print(f"C = {C}: b = {b:.2f}, w = ({w_years:.2f}, {w_pubs:.2f}),"
              f" точност {model.score(tenure[features], tenure['tenured']):.2f},"
              f" P(позиция) за 3 години с позиция: {p_exceptions[0]:.3f}, за 7 години без позиция: {p_exceptions[1]:.3f}")
    for s in [0.1, 0.2, 0.5, 1, 2, 5]:
        value = log_likelihood(-49 * s, [5 * s, 2 * s], tenure[features], tenure["tenured"])
        print(f"s = {s:3}: w = ({5 * s:4.1f}, {2 * s:4.1f}), b = {-49 * s:6.1f}, l(w) = {value:.6f}")


def to_test_6(softmax_model, sepal_test_std):
    # Моделът се колебае между versicolor (0.394) и virginica (0.606).
    # Увеличаването на z на versicolor с 1 умножава e^z по e ≈ 2.7: P(versicolor)
    # расте до около 0.64, а другите вероятности намаляват.
    # Добавянето на едно и също число c към всички z не променя нищо:
    # e^(zₖ + c) = e^c · e^(zₖ), а множителят e^c е и в числителя, и в знаменателя
    # и се съкращава. Важни са само разликите между резултатите.
    z = softmax_model.decision_function(sepal_test_std[1:2])[0]
    z_up = z.copy()
    z_up[1] += 1
    table = pd.DataFrame({
        "z": z, "P": softmax(z), "z, versicolor + 1": z_up, "P след + 1": softmax(z_up),
        "z + 5": z + 5, "P след z + 5": softmax(z + 5),
    }, index=softmax_model.classes_)
    display(table.round(3))


# --- Задача: спам филтър с вероятности ---


def train(data_path=None):
    """SMS съобщенията: Bag-of-Words, логистична регресия, перцептрон и постоянна прогноза."""
    if data_path is None:
        data_path = Path(__file__).resolve().parent / "data" / "sms_spam.csv"
    sms = pd.read_csv(data_path)
    y = (sms["label"] == "spam").astype(int)
    train_texts, test_texts, y_train, y_test = train_test_split(
        sms["text"], y, test_size=0.2, random_state=42, stratify=y
    )
    vectorizer = CountVectorizer(min_df=2).fit(train_texts)
    X_train, X_test = vectorizer.transform(train_texts), vectorizer.transform(test_texts)
    logistic = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    perceptron = Perceptron(random_state=42).fit(X_train, y_train)
    baseline = DummyClassifier().fit(X_train, y_train)
    return {
        "vectorizer": vectorizer,
        "logistic": logistic,
        "X_test": X_test,
        "y_test": y_test,
        "train_size": len(train_texts),
        "test_size": len(test_texts),
        "accuracy": {
            "постоянна прогноза": baseline.score(X_test, y_test),
            "перцептрон": perceptron.score(X_test, y_test),
            "логистична регресия": logistic.score(X_test, y_test),
        },
    }


def thresholds(result, values=(0.2, 0.5, 0.9)):
    """Хванатия спам и нормалните съобщения, отбелязани като спам, при няколко прага."""
    p = result["logistic"].predict_proba(result["X_test"])[:, 1]
    y = result["y_test"].to_numpy()
    rows = []
    for threshold in values:
        spam = p >= threshold
        rows.append({
            "праг": threshold,
            "хванат спам": int((spam & (y == 1)).sum()),
            "пропуснат спам": int((~spam & (y == 1)).sum()),
            "нормални, отбелязани като спам": int((spam & (y == 0)).sum()),
        })
    return pd.DataFrame(rows)


def main():
    result = train()
    vectorizer, logistic = result["vectorizer"], result["logistic"]
    print(f"Тренировъчни: {result['train_size']}, тестови: {result['test_size']},"
          f" думи в речника: {len(vectorizer.vocabulary_)}")
    for name, accuracy in result["accuracy"].items():
        print(f"Тестова точност, {name}: {accuracy:.4f}")

    # Положителен коефициент тегли към спам: думата умножава шансовете по eʷ.
    words = vectorizer.get_feature_names_out()
    coef = logistic.coef_[0]
    order = np.argsort(coef)
    print("\nНай-голям коефициент (към спам):")
    print(", ".join(f"{words[i]} {coef[i]:.2f} (×{np.exp(coef[i]):.1f})" for i in order[::-1][:10]))
    print("Най-малък коефициент (към legit):")
    print(", ".join(f"{words[i]} {coef[i]:.2f}" for i in order[:10]))

    messages = ["Congratulations! You have won a FREE prize. Call now to claim",
                "Are we still meeting for lunch tomorrow?"]
    for message, p in zip(messages, logistic.predict_proba(vectorizer.transform(messages))[:, 1]):
        print(f"\nP(спам) = {p:.3f}: {message}")

    # С по-нисък праг се хваща повече спам, но се губят и нормални съобщения.
    print(f"\nТестови спам съобщения: {int(result['y_test'].sum())}")
    print(thresholds(result).to_string(index=False))


if __name__ == "__main__":
    main()
