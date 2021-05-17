# Deep Learning Based Analysis of Vehicle Mobility and Traffic Patterns for Autonomous Vehicles
This report investigates a potential alternative design to the implementation of RSUs for the prediction of autonomous vehicle traffic patterns. The design comprises of a series of LSTM models that are implemented for multiple intersections. In the problem definition, it is stated that the data must be accurate and real signifying that simulated data generation is not accepted, and the model should successful in predicting the traffic patterns of vehicles. In this solution, the two-design criterion are met – accurate and real data for 6 intersections were gathered and used for training and testing, and the prediction of traffic was achieved with an average accuracy of 97% through the 6 chosen intersections.

When designing the final product, the current solutions regarding traffic prediction was initially researched to understand the options that exist. This research consisted of an analysis on the current infrastructure, environmental, social and economic impact, and current solutions for autonomous vehicles. This was done by finding various scholarly articles from organizations such as Massachusetts Institute of Technology (MIT), The Governor’s Highway Safety Association (GHSA), etc. It was determined that there is a clear lack of infrastructure in the industry as it is in its early stages of development.

After these current solutions were defined, the design and approach for the implementation was investigated. Multiple datasets were collected from the City of Victoria located in Melbourne, Australia. This data met the criterion as it was historical traffic data from 2014 that was maintained by the State of Victoria. The different datasets contained information on date, time, traffic volume, location coordinates, accidents, conditions of road, weather conditions, and etc. Preprocessing was completed to extract the correct features required for the model. An LSTM model was then built to predict the traffic patterns of the intersection located at Elizabeth St and Little Collins St. The model was then expanded to 6 intersection to widen the scope of the predictions.

Finally, a stakeholder analysis was completed alongside additional recommendations on scaling the implemented solution to meet greater needs which includes the effects of seasons and reaching wider range of intersections to create an interconnected network.

Group members:  Ann Fernandes, Luvit Chumber, Sasanka Wickramasinghe, Serena Sinclair

## Setup
This command should work once in you cd into the wrapper folder

virtualenv venv && .\venv\Scripts\activate && pip install -r requirements.txt

## Running
### CMD
1. Cd to Capstone/wrapper
2. python main.py
or .\venv\Scripts\python.exe .\main.py
### PyCharm
1. Create configuration pointing to new venv
