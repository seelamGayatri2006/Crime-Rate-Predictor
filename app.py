from flask import Flask, request, render_template
import pickle
import math
import matplotlib
matplotlib.use('Agg')  # ✅ Fix the GUI error
import matplotlib.pyplot as plt
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("Model/model.pkl", "rb"))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict_result():
    city_names = { '0': 'Ahmedabad', '1': 'Bengaluru', '2': 'Chennai', '3': 'Coimbatore', '4': 'Delhi', '5': 'Ghaziabad', '6': 'Hyderabad', '7': 'Indore', '8': 'Jaipur', '9': 'Kanpur', '10': 'Kochi', '11': 'Kolkata', '12': 'Kozhikode', '13': 'Lucknow', '14': 'Mumbai', '15': 'Nagpur', '16': 'Patna', '17': 'Pune', '18': 'Surat' }

    crimes_names = { '0': 'Crime Committed by Juveniles', '1': 'Crime against SC', '2': 'Crime against ST', '3': 'Crime against Senior Citizen', '4': 'Crime against children', '5': 'Crime against women', '6': 'Cyber Crimes', '7': 'Economic Offences', '8': 'Kidnapping', '9': 'Murder' }

    population = { '0': 63.50, '1': 85.00, '2': 87.00, '3': 21.50, '4': 163.10, '5': 23.60, '6': 77.50, '7': 21.70, '8': 30.70, '9': 29.20, '10': 21.20, '11': 141.10, '12': 20.30, '13': 29.00, '14': 184.10, '15': 25.00, '16': 20.50, '17': 50.50, '18': 45.80 }

    city_code = request.form["city"]
    crime_code = request.form['crime']
    year = int(request.form['year'])
    pop = population[city_code]

    # Adjust population with a 1% annual growth rate
    year_diff = year - 2011
    pop *= (1.01 ** year_diff)

    # Predict crime rate
    crime_rate = model.predict([[year, city_code, pop, crime_code]])[0]

    city_name = city_names[city_code]
    crime_type = crimes_names[crime_code]

    if crime_rate <= 1:
        crime_status = "Very Low Crime Area"
    elif crime_rate <= 5:
        crime_status = "Low Crime Area"
    elif crime_rate <= 15:
        crime_status = "High Crime Area"
    else:
        crime_status = "Very High Crime Area"

    cases = math.ceil(crime_rate * pop)

    # 📊 Generate Crime Rate Graph
    years = np.arange(2011, year + 1)
    predicted_rates = [model.predict([[y, city_code, pop * (1.01 ** (y - 2011)), crime_code]])[0] for y in years]

    plt.figure(figsize=(8, 5))
    plt.plot(years, predicted_rates, marker='o', linestyle='-', color='b', label="Predicted Crime Rate")
    plt.axhline(y=crime_rate, color='r', linestyle='--', label="Current Year Prediction")
    plt.xlabel("Year")
    plt.ylabel("Crime Rate")
    plt.title(f"Crime Rate Prediction for {crime_type} in {city_name}")
    plt.legend()
    plt.grid()

    # Save graph to static folder
    graph_path = os.path.join("static", "crime_plot.png")
    plt.savefig(graph_path)
    plt.close()  # ✅ Prevents Matplotlib from holding memory

    return render_template('result.html', city_name=city_name, crime_type=crime_type, year=year, crime_status=crime_status, crime_rate=crime_rate, cases=cases, population=pop, graph_url="static/crime_plot.png")

if __name__ == "__main__":
    app.run(debug=True, port=8990)
