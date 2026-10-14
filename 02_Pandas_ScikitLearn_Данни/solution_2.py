from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def to_test_1(df):
    # 1. Нека използваме any() за постигане на същото - таблица без липсващи стойности
    df_no_missing_2 = df.loc[~(df.isna().any(axis=1))]
    print("Shape:", df_no_missing_2.shape)
    print("Останали пропуски:", df_no_missing_2.isna().sum().sum())
    # 2. Нека извадим само редовете, които имат празни стойности
    display(df.loc[df.isna().any(axis=1)])

def to_test_2(X, X_filled):
    mask = X.isna().any(axis=1)
    # 1. Да намерим първия ТЕСТОВ ред с липсващо измерване.
    missing_index = X.index[mask][0]
    # 2. Да покажем същия ред преди и след попълването чрез неговия индекс.
    display(X.loc[missing_index])
    display(X_filled.loc[missing_index])
    # 3. Коя колона е променена?
    print("Променена колона:", X.columns[X.loc[missing_index].isna()].tolist())

def to_test_3(X, X_train_cat_filled, categorical_features):
    # 1. Какво ще стане, ако кодираме суровата тренировъчна колона без предварително попълване?
    # 2. Сравняваме категориите и броя one-hot колони с правилния encoder.
    raw_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X[categorical_features])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_train_cat_filled)
    print("Без попълване:", raw_encoder.categories_[0].tolist())
    print("Брой колони без попълване:", len(raw_encoder.get_feature_names_out(categorical_features)))
    print("След попълване:", encoder.categories_[0].tolist())
    print("Брой колони след попълване:", len(encoder.get_feature_names_out(categorical_features)))

def to_test_4(encoder, categorical_features):
    # "изкуствена светлина" не присъства в тренировъчните данни.
    # Преобразуваме я със същия encoder. С какво резултатът се
    # различава от кода на "слънце"? Какво означават нулите?
    new_conditions = pd.DataFrame(
        {"light_conditions": ["слънце", "изкуствена светлина"]}
    )
    display(pd.DataFrame(
        encoder.transform(new_conditions),
        columns=encoder.get_feature_names_out(categorical_features)
    ))

def to_test_5(X_test_minmax):
    # 1. Проверяваме дали в РЕАЛНИТЕ тестови редове има стойности извън интервала [0, 1] след min-max мащабиране.
    # 2. Отпечатваме общия им брой и колоните, в които се срещат, както и самите стойности.
    outside = (X_test_minmax < 0) | (X_test_minmax > 1)
    print("Общ брой стойности извън [0, 1]:", outside.sum().sum())
    print("Колони:", list(outside.columns[outside.any(axis=0)]))
    display(X_test_minmax.where(outside).dropna(how="all"))
    # where(outside) - запазва само стойностите, за които outside е True, а останалите заменя с NaN.
    # dropna(how="all") - премахва редовете, в които няма нито една стойност извън интервала.
    display(X_test_minmax.loc[outside.any(axis=1)]) # принтира целия ред
    print(X_test_minmax.to_numpy()[outside.to_numpy()]) # принтира само стойностите извън диапазона

def to_test_6(X_test_scaled):
    # Намираме коя СТАНДАРТИЗИРАНА тестова колона има средно,
    # най-далече от нулата. Отпечатваме името и средното.
    test_means = X_test_scaled.mean()
    farthest_feature = test_means.abs().idxmax()
    print("Колона:", farthest_feature)
    print("Средно:", test_means[farthest_feature])

def to_test_7(neighbors, y_train):
    # Избираме първия тестов ирис с променен съсед.
    # Намираме вида на двата най-близки тренировъчни ириса чрез y_train.loc.
    first_changed = neighbors.loc[neighbors["changed"]].iloc[0]

    minmax_species = y_train.loc[first_changed["minmax_neighbor"]]
    standard_species = y_train.loc[first_changed["standard_neighbor"]]
    
    print("Тестов ирис:", first_changed["test_index"])
    print("Вид на съседа при MinMaxScaler:", minmax_species)
    print("Вид на съседа при StandardScaler:", standard_species)

    # Различни ли са видовете, или се е сменил само конкретният ред?
    if minmax_species == standard_species:
        print("Сменил се е само конкретният съсед, но не и видът.")
    else:
        print("Сменили са се както съседът, така и видът.")

def as_frame(values, index, columns):
    return pd.DataFrame(values, index=index, columns=columns)

def main():
    data_path = Path(__file__).parent / "data" / "saheart_exercise.csv"
    heart = pd.read_csv(data_path)
    heart_X = heart.drop(columns=["row.names", "chd"])
    heart_y = heart["chd"]
    heart_numeric = heart_X.select_dtypes(include="number").columns.tolist()
    heart_categorical = ["famhist"]

    heart_X_train, heart_X_test, heart_y_train, heart_y_test = train_test_split(
        heart_X, heart_y, test_size=0.2, random_state=42, stratify=heart_y
    )
    heart_num_imputer = SimpleImputer(strategy="median").fit(
        heart_X_train[heart_numeric]
    )
    heart_cat_imputer = SimpleImputer(strategy="most_frequent").fit(
        heart_X_train[heart_categorical]
    )
    heart_train_num = as_frame(
        heart_num_imputer.transform(heart_X_train[heart_numeric]),
        heart_X_train.index, heart_numeric
    )
    heart_test_num = as_frame(
        heart_num_imputer.transform(heart_X_test[heart_numeric]),
        heart_X_test.index, heart_numeric
    )
    heart_train_cat = as_frame(
        heart_cat_imputer.transform(heart_X_train[heart_categorical]),
        heart_X_train.index, heart_categorical
    )
    heart_test_cat = as_frame(
        heart_cat_imputer.transform(heart_X_test[heart_categorical]),
        heart_X_test.index, heart_categorical
    )

    heart_encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    ).fit(heart_train_cat)
    heart_scaler = StandardScaler().fit(heart_train_num)
    encoded_names = heart_encoder.get_feature_names_out(heart_categorical)

    def ready_table(numbers, categories):
        scaled = as_frame(
            heart_scaler.transform(numbers), numbers.index, heart_numeric
        )
        encoded = as_frame(
            heart_encoder.transform(categories), categories.index, encoded_names
        )
        return pd.concat([scaled, encoded], axis=1)

    heart_train_ready = ready_table(heart_train_num, heart_train_cat)
    heart_test_ready = ready_table(heart_test_num, heart_test_cat)
    assert heart_train_ready.columns.equals(heart_test_ready.columns)
    assert not heart_train_ready.isna().any().any()
    assert not heart_test_ready.isna().any().any()

    heart_model = KNeighborsClassifier(n_neighbors=5).fit(
        heart_train_ready, heart_y_train
    )
    heart_predictions = heart_model.predict(heart_test_ready)
    heart_accuracy = accuracy_score(heart_y_test, heart_predictions)

    heart_baseline = DummyClassifier(strategy="most_frequent").fit(
        heart_train_ready, heart_y_train
    )
    baseline_accuracy = accuracy_score(
        heart_y_test, heart_baseline.predict(heart_test_ready)
    )

    case = heart_test_ready.iloc[[4]]
    predicted_chd = int(heart_model.predict(case)[0])
    actual_chd = int(heart_y_test.iloc[4])

    print("Готови таблици:", heart_train_ready.shape, heart_test_ready.shape)
    print("Точност на 5-NN:", round(heart_accuracy, 3))
    print("Точност на базовия модел:", round(baseline_accuracy, 3))
    print("Пети тестов ред:", heart_X_test.index[4])
    print("Прогноза:", predicted_chd, "действителна стойност:", actual_chd)
    print("Допълнителните признаци не гарантират по-висока точност.")


if __name__ == "__main__":
    main()
