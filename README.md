# Capstone1----ML_Zoom_Camp.-2022

This project has been developed with all the penalties from Messi from the firstone in 2005 to the last one against Netherlands during 2022 WC. 

These are the following variables:

  "team_for": (teams where he plays),
  "time": (minute of the penalty),
  "scoreline": (scoreline at the moment of the penalty),
  "venue": (venue of the match),
  "history": (0 if he failed the last penalty or 1 if he scored it),
  "competition": (competition of the match)

Local deployment pip install pipenv pipenv install numpy pandas seaborn bentoml==1.0.7 scikit-learn==1.1.3 xgboost==1.7.1 pydantic==1.10.2 Enter shell. To open the notebook.ipynb and see all the models pipenv shell For the following you need to run train.py pipenv run python train.py Then, get the service running on localhost pipenv run bentoml serve service.py:svc Click on 'Try it out' and make accurate predictions that is guaranteed to blow the minds of your pals. Optional: Run locust to test server, make sure you have installed it pipenv install locust pipenv run locust -H http://localhost:3000 and check it out on browser. Production deployment with BentoML You need to have Docker installed.

First we need to build the bento with

pipenv run bentoml build Docker container Once we have the bento tag we containerize it. pipenv run bentoml containerize messi_model:tag Replace tag with the tag you get from bentoml build

## Testing the model 

<img width="1342" alt="Screenshot 2022-12-13 at 19 59 41" src="https://user-images.githubusercontent.com/98197260/207487263-472d1f34-ee12-4371-9968-8e2777776d9e.png">

## Link to the model:
http://35.175.226.93:3000/#/Service%20APIs/messi_model__classify

