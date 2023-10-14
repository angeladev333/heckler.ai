import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, BaggingRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier

from sklearn.metrics import accuracy_score  # Accuracy metrics
import pickle  # For saving model

df = pd.read_csv('coords.csv')
X = df.drop('class', axis=1)  # features
y = df['class']  # target value

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234)

pipelines = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier(max_iter=1000)),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

# 'bag': make_pipeline(StandardScaler(), BaggingClassifier()),
#     'hgb': make_pipeline(StandardScaler(), HistGradientBoostingClassifier()),
#     'hgbreg': make_pipeline(StandardScaler(), HistGradientBoostingRegressor()),
#     'bagreg': make_pipeline(StandardScaler(), BaggingRegressor()),


fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train.values, y_train.values)
    fit_models[algo] = model

for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))

with open('body_language.pkl', 'wb') as f:
    pickle.dump(fit_models['lr'], f)

# print(fit_models['rf'].predict(X_test))
# print(y_test)
