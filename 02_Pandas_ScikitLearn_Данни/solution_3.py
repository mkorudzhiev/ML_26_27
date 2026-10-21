from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def to_test_1(images, y, show_images):
    # y == 7 е булев масив със True на позициите, където етикетът е 7.
    # Използван като индекс, избира само тези редове от images.
    # Резултатът е масив (150, 28, 28) - всичките 150 седмици.
    # Визуално се вижда, че въпреки еднаквия етикет пикселите са различни -
    # всеки човек пише "7" по различен начин.
    seven_images = images[y == 7]
    print("Брой:", len(seven_images))
    show_images(seven_images[:6], ["Етикет 7"] * 6, ncols=6)

def to_test_2(X, images, y, show_images):
    # X[123] е ред от 784 числа. reshape(28, 28) го подрежда обратно в 28 реда по 28 колони.
    # np.array_equal дава True, защото reshape само пренарежда стойностите, без да ги променя.
    # Пикселът от ред 10, колона 14 е на позиция 28 * 10 + 14 = 294 в X[123].
    restored = X[123].reshape(28, 28)
    print("Съвпада с images[123]:", np.array_equal(restored, images[123]))
    position = 28 * 10 + 14
    print("Позиция на пиксела (10, 14):", position)
    print(f"X[123, {position}] = {X[123, position]} | images[123, 10, 14] = {images[123, 10, 14]}")
    show_images([restored], [f"Етикет {y[123]}"], ncols=1)

def to_test_3(unmatched_positions, test_images, y_test, predictions, show_images):
    # unmatched_positions са позициите в тестовото множество, където predictions != y_test.
    # Една и съща позиция избира картинката, верния етикет и предсказанието,
    # защото test_images, y_test и predictions са подредени по един и същ начин.
    # При random_state=42 позиция 13 (вярно 2, модел 8) е сплескана и наклонена
    # и е трудна и за човек. Позиции 4 и 9 са ясни за човек, но моделът греши.
    positions = unmatched_positions[:6]
    show_images(test_images[positions],
                [f"Вярно: {y_test[p]} | модел: {predictions[p]}" for p in positions],
                ncols=6)

def to_test_4(cm, test_images, y_test, predictions, show_images):
    # Диагоналът съдържа правилните отговори, затова се занулява в копие на cm.
    # cm.copy() е нужно, защото np.fill_diagonal променя масива на място.
    # argmax връща позицията на максимума в "сплескания" масив,
    # а np.unravel_index я превръща в (ред, колона), т.е. (верен, предсказан).
    # При random_state=42 четири двойки са наравно с по 4 грешки: 2 като 8,
    # 3 като 2, 4 като 9 и 9 като 8. argmax връща първата по редове: 2 като 8.
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    worst = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
    print("Най-бъркана двойка: верен", worst[0], "предсказан", worst[1], "|", cm_copy[worst], "пъти")
    positions = np.flatnonzero((y_test == worst[0]) & (predictions == worst[1]))[:3]
    show_images(test_images[positions],
                [f"Вярно: {y_test[p]} | модел: {predictions[p]}" for p in positions],
                ncols=3)

def to_test_5(model):
    # Празното изображение е масив 28x28 от нули. reshape(1, -1) го прави ред от 784 признака.
    # Моделът няма клас "няма цифра". Той винаги изчислява 10 линейни оценки и избира най-голямата.
    # При random_state=42 моделът избира 5.
    blank_image = np.zeros((28, 28))
    #rng = np.random.default_rng(42)
    #positions = rng.choice(784, size=80, replace=False)
    #blank_image.flat[positions] = 1.0
    blank_row = blank_image.reshape(1, -1)
    blank_prediction = model.predict(blank_row)[0]
    scores = model.decision_function(blank_row)[0]
    print("Предсказание за празното изображение:", blank_prediction)
    print("Линейни оценки:", scores.round(1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained",
                                   gridspec_kw={"width_ratios": [1, 2]})
    ax1.imshow(blank_image, cmap="gray_r")
    # Рамката остава видима, иначе празното изображение изобщо не се вижда.
    ax1.set(xticks=[], yticks=[], title=f"Празно изображение\nмодел: {blank_prediction}")
    colors = ["#168c81" if c == blank_prediction else "#bfc8d1" for c in model.classes_]
    ax2.bar(model.classes_, scores, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set(xticks=model.classes_, xlabel="Клас", ylabel="Линейна оценка",
            title="Десетте оценки за празното изображение")
    plt.show()

def train(random_state=42, data_dir=None):
    """Зареждане от PNG, обучение, матрица на объркванията."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data" / "shapes_png"
    data_dir = Path(data_dir)
    paths = sorted(data_dir.rglob("*.png"))
    df = pd.DataFrame({
        "filepath": [str(p) for p in paths],
        "label": [p.parent.name for p in paths],
    })
    images = np.array([plt.imread(str(p)) for p in paths])
    labels = df["label"].values
    X = images.reshape(len(images), -1)

    X_train, X_test, y_train, y_test, _, test_images = train_test_split(
        X, labels, images, test_size=0.2, random_state=random_state, stratify=labels
    )
    model = Perceptron(random_state=random_state).fit(X_train, y_train)
    predictions = model.predict(X_test)
    classes = sorted(set(labels))
    cm = confusion_matrix(y_test, predictions, labels=classes)

    return {
        "model": model,
        "test_images": test_images,
        "y_test": y_test,
        "predictions": predictions,
        "classes": classes,
        "cm": cm,
        "correct": int((predictions == y_test).sum()),
        "total": len(y_test),
    }


def plot_result(result):
    """Матрица на объркванията и грешно разпознати примери."""
    cm = result["cm"]
    classes = result["classes"]
    n = len(classes)

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    im = ax.imshow(cm, cmap="Blues")
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=11)
    ax.set(xticks=range(n), xticklabels=classes,
           yticks=range(n), yticklabels=classes,
           xlabel="Предсказан клас", ylabel="Верен клас",
           title="Матрица на объркванията (фигури)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    wrong = np.flatnonzero(result["predictions"] != result["y_test"])
    positions = wrong[:2]
    if len(positions) > 0:
        fig2, axes = plt.subplots(1, len(positions), figsize=(4 * len(positions), 4),
                                  squeeze=False, layout="constrained")
        for ax, pos in zip(axes.ravel(), positions):
            ax.imshow(result["test_images"][pos], cmap="gray_r", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_title(
                f"Верен: {result['y_test'][pos]}\n"
                f"Модел: {result['predictions'][pos]}",
                fontsize=10)
            ax.axis("off")
        fig2.suptitle("Грешно разпознати примери", fontsize=12)

    return fig


def main():
    result = train()
    print(f"Правилно разпознати: {result['correct']} от {result['total']}")
    print(f"Точност: {result['correct'] / result['total']:.3f}")
    print(f"\nКласове: {result['classes']}")
    print(f"\nМатрица на объркванията:")
    print(result["cm"])
    cm_copy = result["cm"].copy()
    np.fill_diagonal(cm_copy, 0)
    worst = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
    print(f"\nНай-бъркана двойка: верен {result['classes'][worst[0]]}"
          f", предсказан {result['classes'][worst[1]]}")
    plot_result(result)
    plt.show()


if __name__ == "__main__":
    main()
