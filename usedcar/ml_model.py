import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def regression_model(car_brand,model, year, transmission, mileage, fueltype, tax, miles_per_gallon, engine_size):
    df = pd.read_csv(f"{car_brand}.csv",sep=",")
    
    # Strip whitespace from all string columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Create separate encoders for each categorical column
    le_model = LabelEncoder()
    le_transmission = LabelEncoder()
    le_fueltype = LabelEncoder()
    
    df["model"] = le_model.fit_transform(df["model"])
    df["transmission"] = le_transmission.fit_transform(df["transmission"])
    df["fuelType"] = le_fueltype.fit_transform(df["fuelType"])

    X = df.drop(["price"],axis=1)
    y = df["price"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    print(X_test)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    r_model = LinearRegression()
    r_model.fit(X_train,y_train)
    y_pred = r_model.predict(X_test)
    print(r2_score(y_test,y_pred))


    # Transform categorical features using the same encoders
    model_encoded = le_model.transform([model.strip()])[0]
    fueltype_encoded = le_fueltype.transform([fueltype.strip()])[0]
    transmission_encoded = le_transmission.transform([transmission.strip()])[0]

    features_list = [model_encoded, year, transmission_encoded, mileage, fueltype_encoded, tax, miles_per_gallon, engine_size]

    scaled_features = scaler.transform([features_list])

    prediction = r_model.predict(scaled_features)

    return prediction

