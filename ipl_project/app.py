from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load the trained model ──────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ipl.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    model = None
    print("⚠️  ipl.pkl not found. Place your trained model file in the project folder.")


# ── Valid teams (must match training data exactly) ──────────────────────────
TEAMS = [
    'Chennai Super Kings',
    'Delhi Daredevils',
    'Kings XI Punjab',
    'Kolkata Knight Riders',
    'Mumbai Indians',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]


def predict_score(bat_team, bowl_team, runs, wickets, overs, runs_last_5, wickets_last_5):
    """
    Build the one-hot encoded feature vector and predict using the loaded model.
    Feature order: bat_team (8) + bowl_team (8) + runs, wickets, overs,
                   runs_last_5, wickets_last_5  → 21 features total
    """
    temp_array = []

    # One-hot encode batting team
    temp_array += [1 if bat_team == team else 0 for team in TEAMS]

    # One-hot encode bowling team
    temp_array += [1 if bowl_team == team else 0 for team in TEAMS]

    # Numeric features
    temp_array += [runs, wickets, overs, runs_last_5, wickets_last_5]

    feature_vector = np.array([temp_array])          # shape (1, 21)
    return int(model.predict(feature_vector)[0])


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    return render_template('predict.html', teams=TEAMS)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Accepts JSON: { bat_team, bowl_team, runs, wickets, overs,
                    runs_last_5, wickets_last_5 }
    Returns JSON: { success, predicted_score, score_min, score_max, message }
    """
    if model is None:
        return jsonify({'success': False,
                        'error': 'Model not loaded. Make sure ipl.pkl is in the project folder.'}), 500

    data = request.get_json()

    # ── Validation ──────────────────────────────────────────────────────────
    bat_team       = data.get('bat_team', '').strip()
    bowl_team      = data.get('bowl_team', '').strip()
    runs           = data.get('runs')
    wickets        = data.get('wickets')
    overs          = data.get('overs')
    runs_last_5    = data.get('runs_last_5')
    wickets_last_5 = data.get('wickets_last_5')

    if bat_team not in TEAMS:
        return jsonify({'success': False, 'error': f'Invalid batting team: "{bat_team}"'}), 400
    if bowl_team not in TEAMS:
        return jsonify({'success': False, 'error': f'Invalid bowling team: "{bowl_team}"'}), 400
    if bat_team == bowl_team:
        return jsonify({'success': False, 'error': 'Batting and bowling team cannot be the same.'}), 400

    try:
        runs           = float(runs)
        wickets        = float(wickets)
        overs          = float(overs)
        runs_last_5    = float(runs_last_5)
        wickets_last_5 = float(wickets_last_5)
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'All numeric fields must be valid numbers.'}), 400

    if not (5.0 <= overs <= 20.0):
        return jsonify({'success': False, 'error': 'Overs must be between 5 and 20.'}), 400
    if not (0 <= wickets <= 10):
        return jsonify({'success': False, 'error': 'Wickets must be between 0 and 10.'}), 400
    if runs < 0:
        return jsonify({'success': False, 'error': 'Runs cannot be negative.'}), 400

    # ── Prediction ──────────────────────────────────────────────────────────
    try:
        score = predict_score(bat_team, bowl_team, runs, wickets,
                              overs, runs_last_5, wickets_last_5)
        return jsonify({
            'success'        : True,
            'predicted_score': score,
            'score_min'      : score - 5,
            'score_max'      : score + 10,
            'message'        : f'Final Predicted Score is {score - 5} to {score + 10}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
