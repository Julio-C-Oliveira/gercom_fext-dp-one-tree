from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from fedt.app.utils import load_dataset

def load_house_client():
    rng = np.random.default_rng()

    X, y = load_dataset()

    number_of_samples = int((len(X)*25)/100)

    idxs = rng.choice(X.shape[0], size=number_of_samples, replace=False)
    X = X.iloc[idxs]
    y = y.iloc[idxs]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, y_train, X_test, y_test

