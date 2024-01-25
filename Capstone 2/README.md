The dataset for this project downloaded from https://www.kaggle.com/datasets/loveall/appliances-energy-prediction

The dataset contains information about house temprature and humidity measurment in differnet areas
There are 8 main location in the building:
    - kitchen
    - living room
    - laundry room
    - office room
    - bathroom
    - ironing room
    - teenager room
    - parents room
And 2 locations outside the building

EDA shows features correlation and different plots about tempreature and energy use

With this information we could create a model for applience energy use prediction in the building

I've tried 2 differnet models with parametr tuning:
      
      - dicision tree regression
      - xgboost regression
      
With parametr tuning xgboost showed more precise momdel for applience energy use prediction

by launching predict.py you can download test csv file and get energy use for this measurments
