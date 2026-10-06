import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import display


def main():
    # Зареждане на данните
    heart = pd.read_csv("data/saheart_exercise.csv")

    # Анализ на данните
    heart_shape = heart.shape
    print('Колко реда и колони има таблицата:', heart_shape)
    missing_mask = heart.isna()
    print('Кои колони съдържат пропуски:', heart.columns[missing_mask.any()]),
    missing_counts = missing_mask.sum()
    print('По колко има във всяка:')
    display(missing_counts[missing_counts > 0])
    positive_share = heart["chd"].mean()
    print('Какъв дял от всички редове имат chd == 1:', positive_share)
    mean_age_by_chd = heart.groupby("chd")["age"].mean()
    print('Каква е средната възраст age поотделно за chd == 0 и chd == 1:', mean_age_by_chd)
    older_mask = heart["age"] >= 50
    older_share = heart.loc[older_mask, "chd"].mean()
    print('Какъв дял от редовете с age >= 50 имат chd == 1:', older_share)

    # Обучение на модел по най-близки съседи
    heart_X = heart[["age", "tobacco"]]
    heart_y = heart["chd"]
    heart_X_train, heart_X_test, heart_y_train, heart_y_test = train_test_split(
        heart_X, heart_y, test_size=0.2, random_state=42, stratify=heart_y
    )
    heart_model = KNeighborsClassifier(n_neighbors=5)
    heart_model.fit(heart_X_train, heart_y_train)

    # Проверка на резултата
    heart_predictions = heart_model.predict(heart_X_test)
    heart_accuracy = accuracy_score(heart_y_test, heart_predictions)
    heart_baseline = DummyClassifier(strategy="stratified")
    heart_baseline.fit(heart_X_train, heart_y_train)
    baseline_predictions = heart_baseline.predict(heart_X_test)
    baseline_accuracy = accuracy_score(heart_y_test, baseline_predictions)
    print("Точност на 5-NN:", round(heart_accuracy, 3))
    print("Точност на базов модел:", round(baseline_accuracy, 3))
    accuracy_difference = (heart_accuracy - baseline_accuracy) * 100
    print(f"Разлика: {accuracy_difference:.1f} процентни пункта")

    # Прогноза за един тестов ред
    case = heart_X_test.iloc[[4]]
    predicted_chd = int(heart_model.predict(case)[0])
    actual_chd = int(heart_y_test.iloc[0])
    display(case)
    print("Прогноза:", predicted_chd, "действителна стойност:", actual_chd)


if __name__ == '__main__':
    main()
