import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.DataFrame({'age': [25,35, None, 45], 'income': [50,40,30,60], 'city': ['X','Y', 'X', 'Z'], 'target': [0,1, 0,1]})
X = df[['age', 'income', 'city']]; y = df['target']
numeric_features = ['age', 'income']
numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_features = ['city']
cat_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features), ('cat', cat_transformer, cat_features)])
clf = Pipeline([('pre', preprocessor), ('rf', RandomForestClassifier(random_state=42))])
clf.fit(X,y)

print('Preprocessor output shape:', preprocessor.transform(X).shape)
print('Pipeline trained successfully.')
