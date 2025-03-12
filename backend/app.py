from flask import Flask, request, jsonify
from flask_cors import CORS
from routing.summarize_indicators import inference, get_analysis_history

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/check-image', methods=['POST'])
def check_image():
    try:
        data = request.get_json()
        if not data:
            raise ValueError("No JSON data received")
            
        image_url = data.get('imageUrl')
        image_id = data.get('imageId')

        if not image_url:
            return jsonify({
                'status': 'error',
                'message': 'No image URL provided'
            }), 400

        # Get analysis results
        result = inference(image_url, image_id)
        if not result:
            return jsonify({
                'status': 'error',
                'message': 'Analysis failed to produce results'
            }), 500

        # Return standardized response
        return jsonify({
            'status': 'success',
            'message': 'Image analyzed successfully',
            'decision': result['decision'],
        }), 200

    except Exception as e:
        print(f"Error in check_image endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/analysis-history', methods=['GET'])
def get_history():
    try:
        history = get_analysis_history()
        return jsonify({
            'status': 'success',
            'message': 'Analysis history retrieved successfully',
            'results': history
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)