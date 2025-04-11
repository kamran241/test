import joblib

# Load and inspect the model file
model = joblib.load('churn_model.pkl')
print("Loaded object type:", type(model))
print("Loaded object:", model)