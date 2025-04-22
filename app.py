import os
import json
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Path, Request
from fastapi.responses import JSONResponse, FileResponse
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from typing import Optional

app = FastAPI()

load_dotenv()
UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

vision_model = ChatOpenAI(
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY"), 
    model="gpt-4o-mini", max_tokens=100)

class foodData(BaseModel):
    name: str = Field(description='Food Name using indonesian')
    calories_100g: float = Field(description='Food Calories with two decimals in kkal')
    carbs_100g: float = Field(description='Food Carbohydrates with two decimals in gram')
    proteins_100g: float = Field(description='Food Protein with two decimals in gram')
    fats_100g: float = Field(description='Food Fats with two decimals in gram')

class foodClassification(BaseModel):
    isFood: bool = Field(description='Classified as food or not')
    food: Optional[foodData] = Field(default=None, description='Food nutrition information')


def create_nutrition_prompt(image_url):
  """Creates a prompt for the nutritionist model with a given image URL."""
  prompts = [
      SystemMessage(
          """You are a nutritionist. Your task is to evaluate images and provide responses based on the following conditions:
1. If the image depicts food, identify and give the most suitable name for the dish. Then, provide detailed nutrition information for the dish.
2. If the image does not depict food, classify it as 'Not Food' and do not provide any nutrition information.
          """),
      HumanMessage(content=[
                  {"type": "text", "text": "if the image is food, give me the nutrient informations for 100g of this food. else dont give me any information of the food nutrient"},
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": image_url
                          },
                  },
              ] 
      )
  ]
  return prompts

import asyncio

async def getNutritionInfo(request: Request,filename: str):
    """Fetches nutrition information using the vision model asynchronously."""
    try:
        image_url = f"{request.base_url}fetchImage/{filename}"
        prompts = create_nutrition_prompt(image_url)
        structured_llm = vision_model.with_structured_output(foodClassification)
        vision_response = await asyncio.to_thread(vision_model.invoke, prompts)
        schema2 = await asyncio.to_thread(structured_llm.invoke, vision_response.content)
        return schema2.json()
    except Exception as e:
        return {"error": f"An error occurred while processing the image: {str(e)}"}


@app.get("/fetchImage/{filename}")
async def view_image(filename: str):
    file_path = os.path.join(UPLOAD_DIRECTORY, filename)
    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    
    return FileResponse(
        file_path,
        media_type="image/jpeg",
        filename=filename,
        headers={"Content-Disposition": "inline"}
    )

@app.post("/upload-image/{userid}")
async def upload_image(
    request: Request,
    userid: str = Path(..., title="The ID of the user"),
    file: UploadFile = File(...)
):
    fileName = f"{userid}.jpg"
    file_path = os.path.join(UPLOAD_DIRECTORY, fileName)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        img = Image.open(file_path) 
        img = img.resize((445, 250))
        
        img.save(file_path)
        food_classification = await getNutritionInfo(request=request,filename=fileName)
        print(food_classification)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    return JSONResponse(content={
        "userID": userid,
        "classification": json.loads(food_classification)
    })