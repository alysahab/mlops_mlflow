from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow


# load dataset
brc = load_breast_cancer()

X = pd.DataFrame(brc.data, columns=brc.feature_names)
y = pd.Series(brc.target, name='target')

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=42)

params = {

    "n_estimators": [10,50,100],
    "max_depth": [None, 10,20,30]
}

rf = RandomForestClassifier()
grid = GridSearchCV(rf, params, cv=5, n_jobs=-1,scoring="f1", verbose=2)


mlflow.set_experiment("hyperparam_tune")
with mlflow.start_run():
    grid.fit(X_train, y_train)
    for i in range(len(grid.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
                mlflow.log_params(grid.cv_results_['params'][i])
                mlflow.log_metric("f1", grid.cv_results_["mean_test_score"][i])

    best_pram = grid.best_params_
    best_score = grid.best_score_

    train_df = X_train.copy()
    train_df['target'] = y_train.values
    test_df = X_test.copy()
    test_df['target'] = y_test.values

    train_df = mlflow.data.from_pandas(train_df)
    test_df = mlflow.data.from_pandas(test_df)

    mlflow.sklearn.log_model(grid.best_estimator_)
    mlflow.log_input(train_df, "training")
    mlflow.log_input(test_df, "testing")
    mlflow.log_params(best_pram)
    mlflow.log_metric("F1", best_score)
