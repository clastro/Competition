import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Load Data 
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# Data Preprocessing
# SUBCLASS가 범주형이기 때문에 LabelEncoder 사용
le_subclass = LabelEncoder()
train['SUBCLASS'] = le_subclass.fit_transform(train['SUBCLASS'])

# Feature and target separation
X = train.drop(columns=['SUBCLASS', 'ID'])
y_subclass = train['SUBCLASS']

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object', 'category']).columns

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'  # Keep the rest of the columns as they are
)

# Define and train model
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# Create and fit pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

pipeline.fit(X, y_subclass)

# Inference
test_X = test.drop(columns=['ID'])
predictions = pipeline.predict(test_X)
original_labels = le_subclass.inverse_transform(predictions)

# Submission
submission = pd.read_csv("./data/sample_submission.csv")
submission["SUBCLASS"] = original_labels
submission.to_csv('./one_hot_baseline_submission.csv', encoding='UTF-8-sig', index=False)
