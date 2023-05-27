import joblib
import pandas as pd
import hotel_pipeline as hp
hotel_reserv = pd.read_csv("data/Hotel Reservations.csv")
df = hotel_reserv.copy()
X, y = hp.booking_stat_data_prep(df)
new_X, low_importance_features = hp.low_impoortance(X, y)
random_user = (new_X.sample(1, random_state=12))
df = pd.concat([df, random_user], ignore_index=True)
new_X, low_importance_features = hp.low_impoortance(X, y)
last_user = new_X.iloc[-1].values.reshape(1, -1)
new_model = joblib.load("final_modell.pkl")
prediction = new_model.predict(last_user)
print(prediction)
