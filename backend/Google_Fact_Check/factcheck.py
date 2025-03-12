import os
from flask import Flask, request, render_template, jsonify
from googleapiclient.discovery import build
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()

app = Flask(__name__)
api_key = os.getenv('GOOGLE_API')


factcheck_service = build('factchecktools', 'v1alpha1', developerKey=api_key)
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

def parse_claim_data(result):
    if not result.get('claims'):
        return None
    
    claim = result['claims'][0]
    review = claim.get('claimReview', [{}])[0]
    
    return {
        'rating': review.get('textualRating', 'Unknown'),
        'date': claim.get('claimDate', 'Unknown date'),
        'source_url': review.get('url', ''),
        'publisher': review.get('publisher', {}).get('name', 'Unknown source'),
        'claim_text': claim.get('text', '')
    }

def get_gemini_summary(claim_data):
    if not claim_data:
        return "No fact check data found."
    
    prompt = f"""
    Based on this fact check:
    Claim: {claim_data['claim_text']}
    Rating: {claim_data['rating']}
    Source: {claim_data['publisher']}
    
    Provide a brief, clear summary stating:
    1. If the claim is true or false
    2. A one-sentence explanation why
    3. The source and date
    """
    
    response = model.generate_content(prompt)
    return response.text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    user_query = request.form.get('user_query', '')
    if not user_query:
        return "No query provided!"
    
    try:
        # Get fact check results
        result = factcheck_service.claims().search(query=user_query).execute()
        
        # Parse and summarize
        claim_data = parse_claim_data(result)
        if not claim_data:
            return jsonify({"error": "No fact check found for this claim"})
        
        summary = get_gemini_summary(claim_data)
        
        return render_template('result.html', 
                            summary=summary,
                            claim_data=claim_data)
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)