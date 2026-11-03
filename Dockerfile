From python:3.14-slim

WORKDIR /app

Copy requirements.txt . 
Run pip install --no-cache-dir -r requirements.txt

Copy Iris_Model.pkl .

Copy app.py .

CMD ["python", "app.py"]
