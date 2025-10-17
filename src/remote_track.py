import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub


# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 5
n_estimators = 20



# Create new experiment
dagshub.init(repo_owner='alysahab', repo_name='my-first-repo', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/alysahab/mlops_mlflow.mlflow")

mlflow.set_experiment("remote_exp1")
with mlflow.start_run():
    rf =  RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    model = rf.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)


    mlflow.set_tags({"author":"AdnanAli", "project": "Wine Classification", "release_version":"1.0.0"})

    # Create Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")

    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('n_estimator',n_estimators)
    mlflow.log_param('max_depth',max_depth)


    
print(accuracy)