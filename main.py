from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def regression():
    # File downloaded at https://data.cityofchicago.org/api/views/xguy-4ndq/rows.csv?accessType=DOWNLOAD
    df = pd.read_csv("crimes.csv")
    crimes_per_year = df.groupby('Year').ID.count().reset_index()
    crimes_per_year.drop(crimes_per_year.tail(1).index, inplace=True)

    x = crimes_per_year.Year
    x = x.values.reshape(-1, 1)

    y = crimes_per_year.ID
    y = y.values.reshape(-1, 1)

    plt.scatter(x, y)

    regmodel = linear_model.LinearRegression()
    regmodel.fit(x, y)

    y_predict = regmodel.predict(x)

    plt.plot(x, y_predict)
    plt.savefig("2001-2022.png")

    x_future = np.array(range(2001, 2050))
    x_future = x_future.reshape(-1, 1)

    future_predict = regmodel.predict(x_future)

    plt.scatter(x, y)
    plt.plot(x_future, future_predict)
    plt.savefig("2001-2050.png")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    regression()

