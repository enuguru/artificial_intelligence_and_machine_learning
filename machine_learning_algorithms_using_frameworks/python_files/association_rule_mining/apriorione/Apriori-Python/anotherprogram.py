from itertools import combinations
from operator import itemgetter
import pandas as pd
from time import time


def perform_apriori(data, support_count):

    single_items = (data['items'].str.split(",", expand=True))\
        .apply(pd.value_counts).sum(axis=1).where(lambda value: value > support_count).dropna()

    apriori_data = pd.DataFrame(
        {'items': single_items.index.astype(int), 'support_count': single_items.values, 'set_size': 1})

    data['set_size'] = data['items'].str.count(",") + 1

    data['items'] = data['items'].apply(lambda row: set(map(int, row.split(","))))

    single_items_set = set(single_items.index.astype(int))

    for length in range(2, len(single_items_set) + 1):
        data = data[data['set_size'] >= length]
        d = data['items'] \
            .apply(lambda st: pd.Series(s if set(s).issubset(st) else None for s in combinations(single_items_set, length))) \
            .apply(lambda col: [col.dropna().unique()[0], col.count()] if col.count() >= support_count else None).dropna()
        if d.empty:
            break
        apriori_data = apriori_data.append(pd.DataFrame(
            {'items': list(map(itemgetter(0), d.values)), 'support_count': list(map(itemgetter(1), d.values)),
             'set_size': length}), ignore_index=True)

    return apriori_data


if __name__ == '__main__':
    table = pd.read_csv('Groceries.csv')
    start = time()
    print(perform_apriori(data=table, support_count=500))
    print(time() - start)
