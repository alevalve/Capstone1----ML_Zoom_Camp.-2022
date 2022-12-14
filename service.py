import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

class PenaltyAim(BaseModel):
    team_for: str
    time: int
    scoreline: str
    venue: str
    history: int
    competition: str



model_ref = bentoml.xgboost.get("messi_model:latest")
dv = model_ref.custom_objects["DictVectorizer"]


model_runner = model_ref.to_runner()


svc = bentoml.Service("messi_model", runners=[model_runner])


@svc.api(input=JSON(pydantic_model=PenaltyAim), output=JSON())
async def classify(cristiano_penalty_aim):
    application_data = cristiano_penalty_aim.dict()
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    print(prediction)

    result = prediction[0]

    if result == 2:
        return {"Aim": "right"}
    elif result == 1:
        return {"Aim": "middle"}
    else :
        return {"Aim": "left"}
    