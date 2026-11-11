from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def rss(actual, predicted):
    """Сумата от квадратите на остатъците."""
    return float(np.sum((np.asarray(actual) - np.asarray(predicted)) ** 2))


def to_test_1(X_train, y_train, tv_model):
    # Правата през двата крайни пазара минава точно през тях, но не гледа
    # останалите 148 и има много по-голям RSS.
    # При фиксиран свободен член RSS е парабола спрямо наклона. Най-ниската ѝ
    # точка е при наклона на LinearRegression: по-малък RSS няма.
    tv = X_train["TV"].to_numpy()
    sales = y_train.to_numpy()
    low, high = tv.argmin(), tv.argmax()
    slope_2 = (sales[high] - sales[low]) / (tv[high] - tv[low])
    intercept_2 = sales[low] - slope_2 * tv[low]
    print(f"Права през двата крайни пазара: {intercept_2:.3f} + {slope_2:.4f} · TV,"
          f" RSS = {rss(sales, intercept_2 + slope_2 * tv):.1f}")
    print(f"LinearRegression: {tv_model.intercept_:.3f} + {tv_model.coef_[0]:.4f} · TV,"
          f" RSS = {rss(sales, tv_model.predict(X_train[['TV']])):.1f}")

    slopes = np.linspace(0.035, 0.06, 251)
    rss_values = [rss(sales, tv_model.intercept_ + slope * tv) for slope in slopes]
    print(f"Най-малък RSS при наклон {slopes[np.argmin(rss_values)]:.4f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(slopes, rss_values, color="black")
    ax.axvline(tv_model.coef_[0], color="#eb6834", linestyle="--",
               label="наклон на LinearRegression")
    ax.set(xlabel="наклон (свободният член е β̂₀)",
           ylabel="RSS върху тренировъчните редове",
           title="RSS при различни наклони")
    ax.legend()
    plt.show()


def to_test_2(df, se_slope):
    # df.sample(200, replace=True) избира 200 пазара с връщане: някои пазари
    # се повтарят, а други липсват. Всяка такава извадка играе ролята на други
    # 200 пазара. Наклоните се разсейват около 0.0475 със стандартно
    # отклонение около 0.0028, почти колкото SE = 0.0027 от формулата.
    # Формулата приема еднаква разсейка на грешките, а в т. 12 разсейката
    # расте с прогнозата, затова повторните извадки дават малко повече.
    # Този подход се нарича bootstrap (ISLP, глава 5).
    slopes = []
    for i in range(1000):
        sample = df.sample(len(df), replace=True, random_state=i)
        slopes.append(LinearRegression().fit(sample[["TV"]], sample["sales"]).coef_[0])
    slopes = np.array(slopes)
    center = LinearRegression().fit(df[["TV"]], df["sales"]).coef_[0]
    print(f"Средно на наклоните: {slopes.mean():.4f}")
    print(f"Стандартно отклонение на наклоните: {slopes.std():.4f} | SE от формулата: {se_slope:.4f}")
    inside = np.mean((slopes > center - 2 * se_slope) & (slopes < center + 2 * se_slope))
    print(f"Наклони в интервала β̂₁ ± 2 · SE: {inside:.1%}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(slopes, bins=30, color="#bfc8d1", edgecolor="white")
    for x in (center - 2 * se_slope, center + 2 * se_slope):
        ax.axvline(x, color="#eb6834", linestyle="--")
    ax.set(xlabel="наклон β̂₁", ylabel="брой извадки",
           title="Наклонът при 1000 повторни извадки (пунктир: β̂₁ ± 2 · SE)")
    plt.show()


def to_test_3(X_test, y_test, tv_test_pred):
    # Пазарите, за които моделът прогнозира твърде малко (положителен остатък),
    # имат почти най-големия бюджет за радио, около 48-49 хил. долара. Тези, за
    # които прогнозира твърде много, почти нямат радиореклама, 2-3 хил. долара.
    # Всички са с голям TV бюджет. Моделът само с TV не знае за радиото, а при
    # големи TV бюджети то променя продажбите много.
    table = X_test.copy()
    table["sales"] = y_test
    table["прогноза"] = tv_test_pred
    table["остатък"] = y_test - tv_test_pred
    largest = table["остатък"].abs().sort_values(ascending=False).index[:6]
    display(table.loc[largest].round(2))


def to_test_4(df, features, X_train, X_test, y_train, y_test):
    # Стъпка 1: TV намалява RSS най-много, от 5417 на 2102. Стъпка 2: radio,
    # до 557. Стъпка 3: newspaper намалява RSS съвсем малко, а коригираното R²
    # пада от 0.8962 на 0.8956, затова остават TV и radio.
    # Върху тестовите пазари TV + radio също е малко по-точен от трите бюджета.
    n = len(df)
    y = df["sales"]
    tss = rss(y, np.full(n, y.mean()))
    chosen, remaining = [], list(features)
    best_adjusted = -np.inf
    while remaining:
        step = {}
        for feature in remaining:
            columns = chosen + [feature]
            step[feature] = rss(y, LinearRegression().fit(df[columns], y).predict(df[columns]))
        feature = min(step, key=step.get)
        p = len(chosen) + 1
        adjusted = 1 - (step[feature] / (n - p - 1)) / (tss / (n - 1))
        print(f"Добавяне на {feature}: RSS = {step[feature]:.1f}, коригирано R² = {adjusted:.4f}")
        if adjusted < best_adjusted:
            print("Коригираното R² намалява, изборът спира.")
            break
        best_adjusted = adjusted
        chosen.append(feature)
        remaining.remove(feature)
    print("Избрани предиктори:", chosen)
    for columns in [chosen, list(features)]:
        model = LinearRegression().fit(X_train[columns], y_train)
        error = mean_absolute_error(y_test, model.predict(X_test[columns]))
        print(f"Тестова MAE с {' + '.join(columns)}: {error:.3f}")


def to_test_5(credit, region_model, region):
    # Без колоната West базова става West: свободният член е средният баланс в
    # West (512.31), а коефициентите са разликите на East (+18.69) и South
    # (+6.18) спрямо West. Параметризацията е друга, но прогнозите са същите:
    # и двата модела прогнозират средния баланс на региона на клиента.
    west_base = pd.get_dummies(credit["Region"]).astype(int).drop(columns="West")
    west_model = LinearRegression().fit(west_base, credit["Balance"])
    print("Базова категория West: β̂₀ =", round(west_model.intercept_, 2))
    print(pd.Series(west_model.coef_, index=west_base.columns).round(2))
    print("Еднакви прогнози:", np.allclose(region_model.predict(region), west_model.predict(west_base)))


def to_test_6(df):
    # Без главните ефекти R² пада от 0.929 на 0.803 само защото нулата на radio
    # е преместена с 10. Прогнозите също се променят, а изборът на нула няма
    # нищо общо с данните. Йерархичният модел дава същото R² (0.968) и същите
    # прогнози. Сменя се само коефициентът на TV, защото той е ефектът на TV
    # при radio = 0, а нулата вече е при 10 хил. долара.
    y = df["sales"]
    for shift in (0, 10):
        radio = df["radio"] - shift
        only_product = pd.DataFrame({"TV_radio": df["TV"] * radio})
        hierarchical = pd.DataFrame({"TV": df["TV"], "radio": radio, "TV_radio": df["TV"] * radio})
        product_model = LinearRegression().fit(only_product, y)
        hierarchical_model = LinearRegression().fit(hierarchical, y)
        print(f"radio - {shift}:")
        print(f"  само TV × radio: R² = {product_model.score(only_product, y):.3f},"
              f" първите три прогнози {product_model.predict(only_product)[:3].round(2)}")
        print(f"  йерархичен:      R² = {hierarchical_model.score(hierarchical, y):.3f},"
              f" първите три прогнози {hierarchical_model.predict(hierarchical)[:3].round(2)},"
              f" коефициенти {hierarchical_model.coef_.round(4)}")


def quadratic(frame):
    """horsepower и horsepower² като две колони."""
    return frame[["horsepower"]].assign(horsepower2=frame["horsepower"] ** 2)


def train(random_state=42, data_path=None):
    """Auto: mpg спрямо horsepower с права и с квадратичен член."""
    if data_path is None:
        data_path = Path(__file__).resolve().parent / "data" / "auto.csv"
    raw = pd.read_csv(data_path, na_values="?")
    auto = raw.dropna(subset=["horsepower"]).reset_index(drop=True)

    line_model = LinearRegression().fit(auto[["horsepower"]], auto["mpg"])
    quad_model = LinearRegression().fit(quadratic(auto), auto["mpg"])

    auto_train, auto_test = train_test_split(auto, test_size=0.2, random_state=random_state)
    baseline = DummyRegressor().fit(auto_train[["horsepower"]], auto_train["mpg"])
    test_line = LinearRegression().fit(auto_train[["horsepower"]], auto_train["mpg"])
    test_quad = LinearRegression().fit(quadratic(auto_train), auto_train["mpg"])
    test_errors = {
        "постоянна прогноза": mean_absolute_error(auto_test["mpg"], baseline.predict(auto_test[["horsepower"]])),
        "права": mean_absolute_error(auto_test["mpg"], test_line.predict(auto_test[["horsepower"]])),
        "с квадратичен член": mean_absolute_error(auto_test["mpg"], test_quad.predict(quadratic(auto_test))),
    }
    return {
        "missing": int(raw["horsepower"].isna().sum()),
        "auto": auto,
        "line_model": line_model,
        "quad_model": quad_model,
        "line_r2": line_model.score(auto[["horsepower"]], auto["mpg"]),
        "quad_r2": quad_model.score(quadratic(auto), auto["mpg"]),
        "train_size": len(auto_train),
        "test_size": len(auto_test),
        "test_errors": test_errors,
    }


def plot_result(result):
    """Двете криви и графиките на остатъците, както в лекцията."""
    auto = result["auto"]
    grid = pd.DataFrame({"horsepower": np.linspace(auto["horsepower"].min(), auto["horsepower"].max(), 200)})
    line_fitted = result["line_model"].predict(auto[["horsepower"]])
    quad_fitted = result["quad_model"].predict(quadratic(auto))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout="constrained")
    axes[0].scatter(auto["horsepower"], auto["mpg"], s=12, color="#bfc8d1")
    axes[0].plot(grid["horsepower"], result["line_model"].predict(grid), color="#eb6834", label="права")
    axes[0].plot(grid["horsepower"], result["quad_model"].predict(quadratic(grid)), color="#2a78d6",
                 label="с квадратичен член")
    axes[0].set(xlabel="horsepower", ylabel="mpg", title="Разход на гориво и мощност")
    axes[0].legend()
    for ax, fitted, title in zip(axes[1:], [line_fitted, quad_fitted], ["Остатъци на правата", "Остатъци с квадратичен член"]):
        ax.scatter(fitted, auto["mpg"] - fitted, s=12, color="#bfc8d1")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(xlabel="прогноза", ylabel="остатък", title=title)
    return fig


def main():
    result = train()
    auto = result["auto"]
    line, quad = result["line_model"], result["quad_model"]
    print("Липсващи стойности в horsepower:", result["missing"])
    print("Коли след почистването:", len(auto))
    print(f"Права: mpg = {line.intercept_:.4f} + ({line.coef_[0]:.4f}) · horsepower, R² = {result['line_r2']:.3f}")
    print(f"С квадратичен член: mpg = {quad.intercept_:.4f} + ({quad.coef_[0]:.4f}) · horsepower"
          f" + {quad.coef_[1]:.5f} · horsepower², R² = {result['quad_r2']:.3f}")
    print(f"\nТренировъчни: {result['train_size']}, тестови: {result['test_size']}")
    for name, error in result["test_errors"].items():
        print(f"Тестова MAE, {name}: {error:.2f} mpg")

    # Остатъците на правата по пет групи прогнози: средните им образуват U.
    groups = pd.qcut(line.predict(auto[["horsepower"]]), 5)
    residuals = auto["mpg"] - line.predict(auto[["horsepower"]])
    print("\nСреден остатък на правата по групи прогнози:", residuals.groupby(groups, observed=True).mean().round(2).tolist())
    plot_result(result)
    plt.show()


if __name__ == "__main__":
    main()
