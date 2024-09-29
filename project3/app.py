from flask import Flask, request, render_template
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)

# Simulate customer training data (age, income, spending_score)
training_data = np.array([
    [25, 50000, 60], [30, 60000, 70], [35, 80000, 90], 
    [40, 120000, 50], [23, 40000, 65], [50, 150000, 30]
])

# Apply K-Means Clustering on training data
kmeans = KMeans(n_clusters=3)
kmeans.fit(training_data)

@app.route('/', methods=['GET', 'POST'])
def index():
    cluster = None
    if request.method == 'POST':
        # Get data from form
        age = float(request.form.get('age'))
        income = float(request.form.get('income'))
        spending_score = float(request.form.get('spending_score'))
        
        # Prepare data for clustering
        data = np.array([[age, income, spending_score]])
        
        # Predict the cluster for the new data point
        cluster = kmeans.predict(data)[0]
        
    return render_template('index.html', cluster=cluster)

if __name__ == '__main__':
    app.run(debug=True)
