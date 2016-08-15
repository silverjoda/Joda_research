
Weather prediction:

This reasearch project was made to experiment with local weather prediction
given data from a local weather station (Strahov, block 9).

The following tasks are of interest:

1) Predict the weather (temperature, rainfall, windspeed) for the next 
X hours given the past Z hours of weather variables (temperature, 
air pressure, humidity, cloud coverage, etc...)

2) Find which part of the day is most reliably predicted

3) Find out if the prediction reliability of a specific part of day is
consistent or if it changes day to day.

4) Find out which variables contibute the most to predicting the weather.

Methodology:

The main classifier is a recurrent neural network (LSTM) and is trained on
past data from the same weather station. The prediction is a sequence to
sequence where the input is the past Z hours of weather and the
output is the next X hours. The output prediction is only the local
temperature (and perhaps other things like rainfall and humidity later)
whereas the input variables are all the variables the the weather API 
provides that are useful. Only raw variables are used. 

Useful info:

The weather data can be obtained in an XML format using the following command:
`curl "http://weather.sh.cvut.cz/weather/export.xml"`


