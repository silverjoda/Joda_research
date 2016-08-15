
Weather prediction:

This reasearch project was made to experiment with local weather prediction
given data from a local weather station (Strahov, block 9).

The following tasks are of interest:

1) Predict the weather (next few hours) given the past 12 hours of weather
variables including temperature, air pressure humidity, cloud coverage, etc...

2) Find which part of the day is most reliably predicted

3) Find out if the prediction reliability of a specific part of day is
consistent or if it changes day to day.

4) Find out which variables contibute the most to predicting the weather.

Methodology:

The main classifier is a recurrent neural network (LSTM) and is trained on
past data from the same weather station. The prediction is a sequence to
sequence where the input is the past 12-24 hours of weather and the
output is the next 2-6 hours. The output prediction is only the local
temperature (and perhaps other things like rainfall and pressure later)
whereas the input variables are all the variables the the weather API provides.

Useful info:

The weather data can be obtained in an XML format using the following command:
curl "http://weather.sh.cvut.cz/weather/export.xml"
