Current files 3/15:

src/inspect_data.py >> Check data/columns for Python nuances before preprocessing.

src/preprocess_data.py >> For encoding data for ML pipeline and loading CSV without unnecessary/error columns.

src/train_model.py >> Functions to log and return pipeline after training model and evaluate consistency using cross validation.

src/interpret_model.py >> Reports and plots drivers of churn using feature importance.

data/Churn4500.csv >> Training data on ~4500 client churn data rows

data/Churn2500.csv >> Testing data on ~2500 client churn data rows

data/ChurnDriversChart_3-15.png >> 

Top 3 Churn Drivers:

      feature        importance
      
num__TotalCharges     0.152004

num__MonthlyCharges   0.135583

num__tenure           0.133076

