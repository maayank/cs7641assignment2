import pandas as pd

def load_cancer():
    names = ['id'] + ['fut' + str(i) for i in range(1, 10)] +  ['malignant']
    df = pd.read_csv('data/cancer/breast-cancer-wisconsin.data.cleaned', sep=',', names=names)
    df.drop(columns=['id'], inplace=True)
    df['malignant'] = df['malignant'] == 4
    return df