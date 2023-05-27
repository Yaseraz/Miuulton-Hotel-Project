################################################
# End-to-End Otel Machine Learning Pipeline II
################################################

import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def booking_stat_data_prep(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe.drop("Booking_ID", axis=1, inplace=True)
    dataframe["booking_status"] = dataframe["booking_status"].apply(lambda x: 1 if x == "Not_Canceled" else 0)

    # Separating categorical and numerical columns
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=7, car_th=20)

    # Categorizing number of week nights stayed
    dataframe["no_of_week_nights_category"] = pd.cut(dataframe["no_of_week_nights"],
                                                     bins=[0, 1, 2, 3, 4, np.inf],
                                                     labels=["0-1", "1-2", "2-3", "3-4", "4+"],
                                                     include_lowest=True)

    # Limiting number of children and adults
    dataframe.loc[dataframe["no_of_children"] > 2, "no_of_children"] = 3
    dataframe.loc[dataframe["no_of_adults"] > 3, "no_of_adults"] = 3

    # Combining weekend and week nights stayed
    dataframe["no_of_nights"] = dataframe["no_of_weekend_nights"] + dataframe["no_of_week_nights"]

    # Calculating total, weekend, and weekday prices
    dataframe["total_price"] = dataframe["avg_price_per_room"] * dataframe["no_of_nights"]
    dataframe["weekend_price"] = dataframe["no_of_weekend_nights"] * dataframe["avg_price_per_room"]
    dataframe["weekday_price"] = dataframe["no_of_week_nights"] * dataframe["avg_price_per_room"]

    # Categorizing arrival month
    dataframe["arrival_season"] = dataframe["arrival_month"]
    dataframe["arrival_season"].replace([12, 1, 2], "Winter", inplace=True)
    dataframe["arrival_season"].replace([3, 4, 5], "Spring", inplace=True)
    dataframe["arrival_season"].replace([6, 7, 8], "Summer", inplace=True)
    dataframe["arrival_season"].replace([9, 10, 11], "Autumn", inplace=True)

    # Categorizing room types
    dataframe["room_type_categorized"] = dataframe["room_type_reserved"].astype(str).replace("Room_Type 3", 0) \
        .replace("Room_Type 2", 1).replace("Room_Type 1", 3).replace("Room_Type 5", 4).replace("Room_Type 4", 5) \
        .replace("Room_Type 7", 6).replace("Room_Type 6", 7)

    #hata alıyoruz burda

    dataframe["room_type_categorized"] = dataframe["room_type_categorized"].astype(int)

    dataframe.drop("room_type_reserved", inplace=True, axis=1)

    # Converting column names to uppercase
    dataframe.columns = [col.upper() for col in dataframe.columns]
    # Extract column names by data type and cardinality
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=7, car_th=20)
    # One hot encode categorical features with low cardinality
    ohe_cols = [col for col in dataframe.columns if ((10 >= dataframe[col].nunique() > 2) and
                                                     (col not in ['ROOM_TYPE_CATEGORIZED', 'NO_OF_ADULTS',
                                                                  'NO_OF_CHILDREN', 'ARRIVAL_YEAR',
                                                                  'NO_OF_PREVIOUS_CANCELLATIONS',
                                                                  'NO_OF_SPECIAL_REQUESTS', 'NO_OF_WEEKEND_NIGHTS']))]
    # Bu kolonlar Ordinal Yani 1-2-3 arasında fark var o nedenle encode edilmesi gerekmiyor.
    dataframe = one_hot_encoder(dataframe, ohe_cols)

    # Label encode binary categorical features
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float]
                   and dataframe[col].nunique() == 2 and col != 'BOOKING_STATUS']
    for col in binary_cols:
        label_encoder(dataframe, col)

    # Scale numerical features using MinMaxScaler
    num_cols = [col for col in dataframe.columns if
                dataframe[col].dtype in ['int64', 'float64'] and col != 'BOOKING_STATUS']
    X_scaled = MinMaxScaler().fit_transform(dataframe[num_cols])
    dataframe[num_cols] = pd.DataFrame(X_scaled, columns=dataframe[num_cols].columns)
    # Convert column names to uppercase
    dataframe.columns = [col.upper() for col in dataframe.columns]
    # Remove irrelevant features
    X = dataframe.drop("BOOKING_STATUS", axis=1)
    # Extract target variable
    y = dataframe["BOOKING_STATUS"]
    return X, y


def low_importance(X, y, n=8):
    lgbm_model = LGBMClassifier(colsample_bytree=0.7,
                                max_depth=10,
                                n_estimators=500,
                                num_leaves=100,
                                random_state=17,
                                subsample=0.5)
    lgbm_model.fit(X, y)

    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': lgbm_model.feature_importances_})
    low_importance_features = feature_importance.sort_values(by='importance', ascending=True)['feature'][:n]

    new_X = X.drop(low_importance_features, axis=1)
    return new_X, low_importance_features


def final_model(X, y):
    print("Final Model")
    new_X, low_importance_features = low_impoortance(X, y)
    lgbm_model = LGBMClassifier()
    lgbm_final = lgbm_model.set_params(colsample_bytree=0.7,
                                       max_depth=10,
                                       n_estimators=500,
                                       num_leaves=100,
                                       random_state=17,
                                       subsample=0.5).fit(new_X, y)

    scorings = ["accuracy", "f1", "roc_auc"]

    for scoring in scorings:
        cv_results = cross_validate(lgbm_final, new_X, y, cv=5, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} (LGBMClassifier)")
    return lgbm_final


def main():
    df = pd.read_csv("C:\\Users\\User\Desktop\Project\Hotel Reservations.csv")
    X, y = booking_stat_data_prep(df)
    final_modell = final_model(X, y)
    joblib.dump(final_modell, "final_modell.pkl")
    return final_modell


if __name__ == "__main__":
    print("İşlem başladı")
    main()

