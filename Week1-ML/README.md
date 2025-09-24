# create project folder
mkdir titanic-ml-api
cd titanic-ml-api

# create virtual environment (Windows PowerShell)
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# (macOS/Linux)
# python3 -m venv .venv
# source .venv/bin/activate

# plugins install
pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn jupyter joblib fastapi uvicorn[standard] pydantic

# Run jupyter if needed
jupyter notebook

# Run app
uvicorn app:app --reload

# Test API on : http://127.0.0.1:8000/docs

curl -X POST "http://127.0.0.1:8000/predict" `
  -H "Content-Type: application/json" `
  -d "{\"pclass\":3,\"sex\":0,\"age\":22,\"fare\":7.25}"


