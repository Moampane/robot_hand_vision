import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open("./new_data.pickle", "rb"))

data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# model = RandomForestClassifier()
model = MLPClassifier(
    hidden_layer_sizes=(
        100,
        50,
        10,
    ),
    random_state=1,
    max_iter=15,
)
# model = KNeighborsClassifier(n_neighbors=10)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print(f"{score*100}% of samples were classified correctly!")

f = open("nn_model.p", "wb")
pickle.dump({"model": model}, f)
f.close()
