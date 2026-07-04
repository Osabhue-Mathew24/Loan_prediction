Loan Prediction


A machine learning project that predicts loan approval status (approved/not approved) from applicant financial and demographic data, using Logistic Regression.

Overview

Lenders need a consistent way to evaluate loan applications based on applicant financial history and demographic factors. This project builds a Logistic Regression classifier trained on historical loan application data to predict whether a loan application is likely to be approved (Loan_Status), deployed as an interactive Streamlit app.

Live App Features


Custom-styled title, header, and bank icon image
A "Background of Study" section explaining the purpose of loan prediction directly in the app
Displays the full dataset used for the analysis in an interactive table
Sidebar inputs for Applicant Income, Loan Amount, Coapplicant Income, Dependents, Property Area, Credit History, and Loan Amount Term (numeric fields bounded by the dataset's min/max values)
Shows the user's input as a table before prediction
On clicking "Check your Loan Status," transforms the relevant inputs using the saved scalers/encoder and returns a clear approved/denied result, each with a corresponding icon image



Dataset

The dataset (Loan_Data.csv) includes applicant information such as:


Gender, Married, Dependents, Education, Self_Employed
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
Credit_History, Property_Area
Target: Loan_Status (Y/N)


Loan_ID was dropped as it carries no predictive value.



Data Preprocessing


Cleaned the Dependents column (removed + symbol, converted to numeric)
Handled missing values: numeric columns filled with median, categorical columns filled with mode — but only for columns with less than 30% missing data; columns exceeding that threshold would have been dropped (none were, in this dataset)
Applied StandardScaler to numeric columns with high variance (ApplicantIncome, CoapplicantIncome), saving each fitted scaler separately
Applied LabelEncoder to categorical columns (Gender, Married, Education, Self_Employed, Property_Area, Loan_Status), saving each fitted encoder separately
Selected final feature set: ApplicantIncome, LoanAmount, CoapplicantIncome, Dependents, Property_Area, Credit_History, Loan_Amount_Term



Model


Algorithm: Logistic Regression (scikit-learn)
Train/test split: 80/20, stratified by target class (491 training rows, 123 test rows)


Test set performance:

ClassPrecisionRecallF1-score0 (Not Approved)0.950.550.701 (Approved)0.830.990.90Accuracy0.85

An Honest Limitation

The model performs noticeably better at identifying approved loans (99% recall) than rejected ones (55% recall on the test set). This means it correctly catches almost all loans that should be approved, but misses a substantial portion of loans that should be rejected — likely a result of class imbalance in the training data (approved cases far outnumber rejected ones). This is worth being aware of before using the model as-is for real lending decisions, and would be a good target for improvement (e.g., class weighting, oversampling, or threshold tuning).



Tech Stack


Python
Streamlit — web app interface
Pandas / NumPy — data handling
Scikit-learn — Logistic Regression, StandardScaler, LabelEncoder, train/test split, classification metrics
Seaborn / Matplotlib — used during exploratory analysis
Joblib — saving/loading the trained model, scalers, and encoders
Jupyter Notebook — data exploration and model development



Project Structure

Loan_prediction/
├── app.py                              # Streamlit app

├── Loan_Data.csv                        # Dataset

├── [notebook].ipynb                     # Data exploration and model training

├── loanmodel.pki                        # Trained Logistic Regression model

├── ApplicantIncome_scaler.pki           # Fitted scaler for ApplicantIncome

├── CoapplicantIncome_scaler.pki         # Fitted scaler for CoapplicantIncome

├── Property_Area_encoder.pki            # Fitted encoder for Property_Area

├── Gender_encoder.pki                   # Fitted encoder for Gender

├── Married_encoder.pki                  # Fitted encoder for Married

├── Education_encoder.pki                # Fitted encoder for Education

├── Self_Employed_encoder.pki            # Fitted encoder for Self_Employed

├── Loan_Status_encoder.pki              # Fitted encoder for Loan_Status



How to Run Locally

bash# Clone the repository
git clone https://github.com/Osabhue-Mathew24/Loan_prediction.git
cd Loan_prediction

# Install dependencies
pip install streamlit pandas numpy scikit-learn seaborn joblib

# Run the app
streamlit run app.py



How It Works

Model training (notebook):


Load and clean the loan application dataset
Encode categorical features and scale high-variance numeric features, saving each transformer for reuse
Split data into training and test sets (stratified to preserve class balance)
Train a Logistic Regression model on the processed features
Evaluate performance using precision, recall, and F1-score on the held-out test set


Deployed app:


User enters applicant details (income, loan amount, dependents, property area, credit history, loan term) via the sidebar
Entered values are displayed back to the user as a table
Applicant Income and Coapplicant Income are scaled, and Property Area is encoded, using the saved transformers
On clicking "Check your Loan Status," the processed input is passed to the trained model
The app displays a clear "approved" or "denied" message with a corresponding icon


Future Improvements


Address class imbalance (e.g., class weighting in Logistic Regression, SMOTE oversampling) to improve recall on the minority class
Compare against other classifiers (Random Forest, XGBoost) to see if performance improves
Add cross-validation for a more robust performance estimate
Display model confidence/probability alongside the approved/denied decision in the app
Deploy the app publicly (e.g., Streamlit Community Cloud) and link it here


Author

Matthew Osabhue Iyasele

Email: mathewiyasele@gmail.com

