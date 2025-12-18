import numpy as np
import pandas as pd



def features(car_brand):

    df = pd.read_csv(f"{car_brand}.csv",sep=",")
    model = df["model"].unique().tolist()
    transmission = df["transmission"].unique().tolist()
    fueltype = df["fuelType"].unique().tolist()
    engine_size = df["engineSize"].unique().tolist()

    return {"car_brand":car_brand,"model":model,"transmission":transmission,"fueltype":fueltype,"engine_size":engine_size}

