from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np

def select_and_train_model(X, y=None, task_type="classification"):
    if task_type in ["classification", "regression"]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Naive Bayes": GaussianNB()
        }

        best_model = None
        best_score = 0
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {acc:.4f}")
            if acc > best_score:
                best_score = acc
                best_model = model

        print(f"\n✅ Best Model: {type(best_model).__name__} with Accuracy: {best_score:.4f}")
        return best_model

    elif task_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor()
        }

        best_model = None
        lowest_error = float("inf")
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"{name} MSE: {mse:.4f}")
            if mse < lowest_error:
                lowest_error = mse
                best_model = model

        print(f"\n✅ Best Model: {type(best_model).__name__} with MSE: {lowest_error:.4f}")
        return best_model

    elif task_type == "clustering":
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X)
        print("\n✅ KMeans clustering completed.")
        return model

    elif task_type == "association":
        print("\n⚠️ Association Rule Mining coming soon (like Apriori or FPGrowth)")
        return None

    else:
        raise ValueError("Unsupported task type.")
