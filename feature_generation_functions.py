import pandas as pd
import spacy
import string as py_string


def get_df_with_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df_new = pd.DataFrame()
    for s in cols:
        df_new[s] = df[s]
    return df_new


def rename_columns(df: pd.DataFrame, col_old_name: str, col_new_name: str) -> pd.DataFrame:
    df = df.rename(columns={col_old_name: col_new_name})
    return df.copy()


def remove_empty_description(df: pd.DataFrame, col_name: str = 'Temat Pracy') -> pd.DataFrame:
    """
    Remove rows with empty description
    """
    df = df[df[col_name].notnull()]
    df = df[df[col_name].str.len() > 1]
    return df


def tokenize_column(df: pd.DataFrame, nlp, col_name: str = 'Opis Pracy',
                    name_for_tokenized_col: str = 'tokens') -> pd.DataFrame:
    """
    Tokenize the description
    """
    df[name_for_tokenized_col] = df[col_name].apply(lambda row: [token.text for token in nlp(row)])
    # descriptions['Cleaned Tokens'] = descriptions['Opis Pracy'].apply(
    #     lambda x: [token.text.translate(polish_to_english) for token in nlp(x)])

    return df


def lower_all_letters(df: pd.DataFrame, col_name: str = 'Opis Pracy') -> pd.DataFrame:
    """
    Change all letters to lower case
    """
    df[col_name] = df[col_name].apply(lambda row: [token.lower() for token in row])
    return df


def change_polish_letters(df: pd.DataFrame, col_name: str = 'Opis Pracy') -> pd.DataFrame:
    """
    Change polish letters to english ones
    """
    polish_to_english = str.maketrans({
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n',
        'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z'
    })
    df[col_name] = df[col_name].apply(lambda row: [token.translate(polish_to_english) for token in row])
    return df


def remove_stop_words(df: pd.DataFrame, nlp, col_name: str = 'Opis Pracy') -> pd.DataFrame:
    """
    Remove stop words from the description

    pass the nlp object from spacy (nlp = spacy.load('pl_core_news_sm'))
    """
    df[col_name] = df[col_name].apply(lambda row: [token for token in row if token not in nlp.Defaults.stop_words])
    return df


def remove_punctuation(df: pd.DataFrame, nlp, col_name: str = 'Opis Pracy') -> pd.DataFrame:
    """
    Remove punctuation from the description
    """
    # nlp = spacy.load('pl_core_news_sm')
    df[col_name] = df[col_name].apply(lambda row: [token for token in row if
                                                   token not in py_string.punctuation])
    return df


def remove_special_characters(df: pd.DataFrame, col_name: str = 'Opis Pracy') -> pd.DataFrame:
    """
    Remove special characters from the description
    """
    special = ['\n', '\t', '\r']
    df[col_name] = df[col_name].apply(lambda row: [token for token in row if token not in special])
    return df


def lematize_column(df: pd.DataFrame, nlp, col_name: str = 'Opis Pracy') -> pd.DataFrame:
    """
    Lematize the description
    """
    df[col_name] = df[col_name].apply(lambda row: ' '.join(row))

    df[col_name] = df[col_name].apply(lambda row: [token.lemma_ for token in nlp(row)])
    return df