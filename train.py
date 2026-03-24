import pandas as pd
from sklearn.svm import SVC
import pickle
import os

file_path = r"C:\Users\anshu\OneDrive\Desktop\Disease pridiction project\datasets\Training.csv"

try:
    data = pd.read_csv(file_path)
    X = data.drop('prognosis', axis=1)
    y = data['prognosis']

    svc = SVC(kernel='linear')
    svc.fit(X, y)

    if not os.path.exists('models'): os.makedirs('models')
    pickle.dump(svc, open('models/svc.pkl', 'wb'))
    print("DONE: svc.pkl file successfully ban gayi hai!")

except Exception as e:
    print(f"Abhi bhi error hai: {e}")