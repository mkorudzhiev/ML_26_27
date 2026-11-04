import itertools

import numpy as np
import pandas as pd
from IPython.display import display

ROUNDS = 100
SITUATIONS = ["начало", "CC", "CD", "DC", "DD"]


def tft_last_move_d(my, their):
    """Tit for Tat, но в последния ход на мача играе D."""
    if len(my) == ROUNDS - 1:
        return "D"
    return their[-1] if their else "C"


def soft_majority(my, their):
    """C, докато другият има поне толкова C, колкото D."""
    return "C" if their.count("C") >= their.count("D") else "D"


def ranking(games, points):
    """Средни точки на ход и спечелени мачове за всеки играч."""
    games = games.assign(
        score=[points(a, b) / ROUNDS for a, b in zip(games["moves"], games["opp_moves"])],
        won=[points(a, b) > points(b, a) for a, b in zip(games["moves"], games["opp_moves"])])
    return (games.groupby("player")
                 .agg(points_per_move=("score", "mean"), matches_won=("won", "sum"))
                 .sort_values("points_per_move", ascending=False)
                 .round(3))


def to_test_1(strategies, tournament, points):
    """Две нови стратегии в турнира на известните стратегии."""
    for name, strategy in [("Soft Majority", soft_majority), ("TFT, D в последния ход", tft_last_move_d)]:
        players = {**strategies, name: strategy}
        table = ranking(tournament(players, players, matches=5), points)
        place = list(table.index).index(name) + 1
        print(f"{name}: място {place} от {len(table)}, "
              f"{table.loc[name, 'points_per_move']} точки на ход "
              f"(Tit for Tat: {table.loc['Tit for Tat', 'points_per_move']})")
    print()
    print("Soft Majority прощава, докато другият е сътрудничил поне в половината ходове.")
    print("Класира се високо, но под Tit for Tat: наказва по-късно и прощава по-трудно.")
    print("'TFT, D в последния ход' използва, че мачът има точно 100 рунда: последното")
    print("предателство не може да бъде наказано. Затова във втория турнир на Axelrod")
    print("дължината на мача е била случайна и неизвестна за участниците.")


def to_test_2(data):
    """Дял на всяка ситуация в примерите срещу всеки противник."""
    shares = pd.crosstab(data["opponent"], data["situation"], normalize="index")[SITUATIONS]
    display(shares.round(2))
    print("Срещу Always C има само ситуации CC и DC, а срещу Always D само CD и DD:")
    print("ходът на другия винаги е един и същ. Grudger предава до края след първото D")
    print("на монетата, затова почти всички примери срещу него са CD и DD. Срещу")
    print("Tit for Two Tats преобладават CC и DC, защото тя прощава едно D.")
    print("Средното за една ситуация смесва примери от различни противници. Моделът се")
    print("учи срещу точно тази смес и правилото му е най-добро срещу нея.")


def to_test_3(rule, from_rule, tournament_score, rng, seeds=(7, 8, 9, 10, 11)):
    """Всички 32 правила, които гледат само предишния рунд, в проверката от точка 22."""
    names = {"C C D C D": "Tit for Tat", "C C D D C": "Pavlov", "C C D D D": "същото като Grudger",
             "C C C C C": "Always C", "D D D D D": "Always D",
             " ".join(rule[s] for s in SITUATIONS): "правилото на модела"}
    rows = []
    
    for moves in itertools.product("CD", repeat=5): # 2 ** 5 = 32 правила
        strategy = from_rule(dict(zip(SITUATIONS, moves)))
        scores = []
        for seed in seeds:
            # Random и Joss в тетрадката теглят от rng; всяко правило започва от едно и също
            # състояние на генератора, както с np.random.default_rng(seed) в точка 22
            rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
            scores.append(tournament_score(strategy).mean())
        rows.append({**dict(zip(SITUATIONS, moves)), "точки на ход": np.mean(scores),
                     "стратегия": names.get(" ".join(moves), "")})
    table = pd.DataFrame(rows).sort_values("точки на ход", ascending=False).reset_index(drop=True)
    table.index = pd.RangeIndex(1, len(table) + 1, name="място")
    display(table.head(6).round(3))
    place = table.index[table["стратегия"] == "правилото на модела"][0]
    print(f"Правила: {len(table)}. Правилото на модела е на място {place}, "
          f"средно от {len(seeds)} повторения на проверката.")
    print()
    print("Сред правилата, които гледат само предишния рунд, няма по-добро срещу тези")
    print("противници. Tit for Tat е второ с малка разлика, която идва от мачовете срещу")
    print("Joss, Always D и Random.")
    print()
    print("Това не значи, че по-добра стратегия няма. Срещу Alternator, който редува C и D,")
    print("най-доброто е D във всеки ход (3 точки на ход), а моделът събира 2.48. За да")
    print("познае Alternator, стратегията трябва да вижда поне два рунда назад. Най-доброто")
    print("правило зависи и от противниците: ако всички бяха Always C, щеше да е 'винаги D'.")


def to_test_4(random_move, tournament, decisions, strategies, sizes=(10, 50, 200, 500), repeats=10):
    """Колко пъти се получава едно и също правило при различен брой мачове."""
    rows = []
    for size in sizes:
        rules = []
        for _ in range(repeats):
            records = tournament({"Random": random_move}, strategies, matches=size)
            averages = decisions(records).pivot_table(index="situation", columns="move",
                                                       values="future", aggfunc="mean")
            rule = averages.idxmax(axis=1)
            rules.append(" ".join(f"{s}:{rule[s]}" for s in SITUATIONS))
        counts = pd.Series(rules).value_counts()
        rows.append({"мачове срещу всеки противник": size,
                     "най-честото правило": counts.index[0],
                     f"пъти от {repeats}": counts.iloc[0],
                     "различни правила": len(counts)})
    display(pd.DataFrame(rows).set_index("мачове срещу всеки противник"))
    print("С малко мачове правилото се сменя от опит на опит: разликите между C и D")
    print("са части от точката, а средните от малко примери са шумни. С повече мачове")
    print("опитите дават все по-често едно и също правило, това от точка 21.")
    print("Колко данни са нужни, зависи от това колко малки са разликите, които")
    print("моделът трябва да различи.")
