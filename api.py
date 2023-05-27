import pandas as pd
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from joblib import load
import hotel_pipeline as hp
from tkinter import messagebox

app = FastAPI(debug=True)
templates = Jinja2Templates(directory="template")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("booking.html", {"request": request})


@app.get("/predict/")
async def makeprediction(request: Request, no_of_adults: int, no_of_children: int, no_of_weekend_nights: int,
                   no_of_week_nights: int, type_of_meal_plan: str, required_car_parking_space: int,
                   room_type_reserved: str, lead_time: int, arrival_year: int, arrival_month: int,
                   arrival_date: int, market_segment_type: str, repeated_guest: int,
                   no_of_previous_cancellations: int, no_of_previous_bookings_not_canceled: int,
                   avg_price_per_room: float, no_of_special_requests: int):
    """
    :param request: request
    :param no_of_adults: integer
    :param no_of_children: integer
    :param no_of_weekend_nights: integer
    :param no_of_week_nights: integer
    :param type_of_meal_plan: str
    :param required_car_parking_space: integer
    :param room_type_reserved: str
    :param lead_time: integer
    :param arrival_year: integer
    :param arrival_month: integer
    :param arrival_date: integer
    :param market_segment_type: str
    :param repeated_guest: integer
    :param no_of_previous_cancellations: integer
    :param no_of_previous_bookings_not_canceled: integer
    :param avg_price_per_room: float
    :param no_of_special_requests: integer
    :return:
    """
    test_data = pd.DataFrame({"no_of_adults": no_of_adults,
                              "no_of_children": no_of_children,
                              "no_of_weekend_nights": no_of_weekend_nights,
                              "no_of_week_nights": no_of_week_nights,
                              "type_of_meal_plan": type_of_meal_plan,
                              "required_car_parking_space": required_car_parking_space,
                              "room_type_reserved": room_type_reserved,
                              "lead_time": lead_time,
                              "arrival_year": arrival_year,
                              "arrival_month": arrival_month,
                              "arrival_date": arrival_date,
                              "market_segment_type": market_segment_type,
                              "repeated_guest": repeated_guest,
                              "no_of_previous_cancellations": no_of_previous_cancellations,
                              "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
                              "avg_price_per_room": avg_price_per_room,
                              "no_of_special_requests": no_of_special_requests}, index=[0])
    hotel_reserve = pd.read_csv("data/Hotel Reservations.csv")
    data = hotel_reserve.copy()
    df = pd.concat([data, test_data], ignore_index=True)
    x, y = hp.booking_stat_data_prep(df)
    new_x, low_importance_features = hp.low_importance(x, y)
    last_user = new_x.iloc[-1].values.reshape(1, -1)
    clf = load("final_modell.pkl")
    prediction = clf.predict(last_user)
    if prediction[0] == 1:
        ID = np.random.randint(1000, 10000, 1)
        messagebox.showinfo(f"Prediction:", f"Booking ID={ID}" + " will be confirmed!")
        return RedirectResponse("/static/PredictionResponse11.html")
    else:
        ID = np.random.randint(1000, 10000, 1)
        messagebox.showinfo(f"Prediction:", f"Booking ID={ID}" + " will be canceled!")
        return RedirectResponse("/static/PredictionResponse12.html")