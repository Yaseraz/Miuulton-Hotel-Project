# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function

from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

########################
# Model Prep
########################

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Model
from lightgbm import LGBMClassifier

pd.set_option("display.max_column", None)
pd.set_option("display.float_format", lambda x: "%3.f" % x)

# Help Kit
from HelpKit import utils

#################################
# Load Dataframe
#################################

hotel_reserv = pd.read_csv("data/Hotel Reservations.csv")
df = hotel_reserv.copy()

df.drop("Booking_ID", axis=1, inplace=True)
df["booking_status"] = df["booking_status"].apply(lambda x: 1 if x == "Not_Canceled" else 0)

#######
# EDA
#######
utils.check_df(df)

for col in df.columns:
    print(col)
    print(df[col].unique())

df["lead_time"].mean()
df["total_price"] = df["avg_price_per_room"] * df["no_of_nights"]
df.info()
df["room_type_reserved"].value_counts()
df.groupby("room_type_reserved").agg({"total_price": "mean"})

df["avg_price_per_room"].mean()

#######
# ?
df.groupby("no_of_special_requests").agg({"total_price": "mean"})
#######
# df.groupby("type_of_meal_plan").agg({"avg_price_per_room": "mean"})
# Ortalama oda fiyatı= 103.42 label encode işlemi
# Meal Plan 1= Standart
# Meal Plan 2= Ekleme yapmış olanlar
# Not Selected= Herhangi bir şey seçmemiş


cat_cols, num_cols, cat_but_car = utils.grab_col_names(df, cat_th=7, car_th=20)

# for col in num_cols:
# eda.num_summary(df, col, plot=True)

df[num_cols].describe().T

for col in cat_cols:
    utils.cat_summary(df, col, plot=False)

    # for col in num_cols:
    utils.target_summary_with_num(df, "booking_status", col)

    # for col in cat_cols:
    utils.target_summary_with_cat(df, "booking_status", col)


# Room_type_reserved?
# Veri seti dengesiz
# Yüzde 67'si 1 yani not_canceled(iptal etmemiş)
# Yüzde 33'ü 0 yani canceled(iptal etmiş)
#    BOOKING_STATUS  Ratio
# 1           24390     67
# 0           11885     33


def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(8, 10)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 7}, linecolor='w', cmap='RdBu')
    plt.show(block=True)


# correlation_matrix(df, num_cols)

# Outlier Analysis
utils.check_outlier(df, num_cols)
# eda.boxplot(df)

# Na
df.isnull().sum()

# Numerik değişkenlerin dağılımlarına ve boxplotlarına bak

df.info()

######################################################
# Feature
######################################################

cat_cols, num_cols, cat_but_car = utils.grab_col_names(df, cat_th=7, car_th=20)

df["no_of_week_nights_category"] = pd.cut(df["no_of_week_nights"],
                                          bins=[0, 1, 2, 3, 4, np.inf],
                                          labels=["0-1", "1-2", "2-3", "3-4", "4+"],
                                          include_lowest=True)

df.loc[df["no_of_children"] > 2, "no_of_children"] = 3
df.loc[df["no_of_adults"] > 3, "no_of_adults"] = 3

df["no_of_nights"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]

df["total_price"] = df["avg_price_per_room"] * df["no_of_nights"]
df["weekend_price"] = df["no_of_weekend_nights"] * df["avg_price_per_room"]
df["weekday_price"] = df["no_of_week_nights"] * df["avg_price_per_room"]

df["arrival_month"].replace([12, 1, 2], "Winter", inplace=True)
df["arrival_month"].replace([3, 4, 5], "Spring", inplace=True)
df["arrival_month"].replace([6, 7, 8], "Summer", inplace=True)
df["arrival_month"].replace([9, 10, 11], "Autumn", inplace=True)

yaz_sezonu_fiyatlar = df[df["arrival_month"] == "Summer"]. \
    groupby("room_type_reserved")["avg_price_per_room"].mean()

kış_sezonu_fiyatlar = df[df["arrival_month"] == "Winter"].groupby("room_type_reserved")["avg_price_per_room"].mean()

sonbahar_fiyatlar = df[df["arrival_month"] == "Autumn"].groupby("room_type_reserved")["avg_price_per_room"].mean()

ilkbahar_fiyat = df[df["arrival_month"] == "Spring"].groupby("room_type_reserved")["avg_price_per_room"].mean()

df["room_type_categorized"] = df["room_type_reserved"].astype(str).replace("Room_Type 3 ", 0).replace("Room_Type 2",
                                                                                                      1).replace(
    "Room_Type 1", 3) \
    .replace("Room_Type 5", 4).replace("Room_Type 4", 5).replace("Room_Type 7", 6).replace("Room_Type 6", 7)

df.columns = [col.upper() for col in df.columns]
##################################
# Encode İşlemleri
##################################

cat_cols, num_cols, cat_but_car = utils.grab_col_names(df, cat_th=7, car_th=20)

# One hot Encoder
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = utils.one_hot_encoder(df, ohe_cols)

# Label Encoder
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2 and col != 'BOOKING_STATUS']

for col in binary_cols:
    utils.label_encoder(df, col)

cat_cols, num_cols, cat_but_car = utils.grab_col_names(df, cat_th=7, car_th=20)

df.info()
# Scale işlemi


num_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'BOOKING_STATUS']
X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

df.columns = [col.upper() for col in df.columns]
##################################
# Base Model
##################################
y = df["BOOKING_STATUS"]
X = df.drop("BOOKING_STATUS", axis=1)


def base_models(X, y, scorings=["roc_auc", "f1", "accuracy"]):
    print("Base Models....")
    classifiers = [('LightGBM', LGBMClassifier())]

    for name, classifier in classifiers:
        for scoring in scorings:
            cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring)
            print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


base_models(X, y, scorings=["roc_auc", "f1", "accuracy", "precision", "recall"])

lgbm_model = LGBMClassifier(random_state=17).fit(X, y)
lgbm_model.get_params()

##################################
# Base Model Validation
##################################
cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.8841902136457614
cv_results['test_f1'].mean()
# 0.9158128983364131
cv_results['test_roc_auc'].mean()


# 0.9458120590881972

##################################
# Base Model Feature İmportance
##################################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(20, 20))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(lgbm_model, X)


##########################################
# Hyperparameter optimization
##########################################

def hyperparameter_optimization(X, y):
    lgbm_model = LGBMClassifier(random_state=17)
    lgbm_params = {"learning_rate": [0.001, 0.01, 0.1],
                   "n_estimators": [100, 200, 300, 400, 500],
                   "max_depth": [-1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "subsample": [0.5, 0.7, 0.9],
                   "num_leaves": [31, 50, 100],
                   "colsample_bytree": [0.5, 0.7, 1]}

    lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

    lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

    cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

    return cv_results['test_f1'].mean()


# AFTER H.PARAMETRE:
# Fitting 5 folds for each of 4050 candidates, totalling 20250 fits
# Out[56]: 0.9272783019848034


lgbm_params = {"learning_rate": [0.001, 0.01, 0.1],
               "n_estimators": [100, 200, 300, 400, 500],
               "max_depth": [-1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               "subsample": [0.5, 0.7, 0.9],
               "num_leaves": [31, 50, 100],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

######################
# Final Model
#####################
# Özellikleri belirleyin
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': lgbm_model.feature_importances_})
low_importance_features = feature_importance.sort_values(by='importance', ascending=True)['feature'][:30]
# En az etkili özellikleri kaldırın
y = df["BOOKING_STATUS"]
new_X = X.drop(low_importance_features, axis=1)

lgbm_final = lgbm_model.set_params(colsample_bytree=0.7, max_depth=10, n_estimators=500,
                                   num_leaves=100, random_state=17, subsample=0.5).fit(new_X, y)

########################
# Model Validation
#########################
cv_results = cross_validate(lgbm_final, new_X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_f1'].mean()
# 0.9264540333020193
cv_results['test_accuracy'].mean()
# 0.8994073053066851
cv_results['test_roc_auc'].mean()


# 0.9545806530846125
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(20, 20))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(lgbm_final, new_X)

######################################################
# 6. Prediction for a New Observation
######################################################
import joblib

random_user = new_X.sample(5, random_state=12)
lgbm_final.predict(random_user)

joblib.dump(lgbm_final, "lgbm_final1.pkl")

new_model = joblib.load("lgbm_final1.pkl")
new_model.predict(random_user)

hotel_reserv[21123]

# 21123
# 22752
# 11020
# 33170
# 18122

# [1, 1, 0, 1, 1]
