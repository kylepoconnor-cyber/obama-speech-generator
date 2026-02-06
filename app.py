"""
Flask Web App for Obama Speech Generator
Simple web interface for generating Obama-style statements
"""

from flask import Flask, render_template, request, jsonify
import os
from obama_generator_pinecone import ObamaRAGGeneratorPinecone
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# Rate limiting: 20 requests per hour per IP address
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["20 per hour"],
    storage_uri="memory://"
)

# Initialize the generator (this happens once when the app starts)
print("Initializing Obama Speech Generator...")
try:
    generator = ObamaRAGGeneratorPinecone()
    print("✓ Generator ready!")
except Exception as e:
    print(f"ERROR initializing generator: {e}")
    generator = None

@app.route('/')
def home():
    """Main page"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
@limiter.limit("10 per hour")  # Extra strict limit for generation (costs money)
def generate():
    """Generate a statement based on user input"""
    if not generator:
        return jsonify({
            'error': 'Generator not initialized. Check your API keys.'
        }), 500
    
    try:
        # Get parameters from request
        data = request.json
        topic = data.get('topic', '')
        length = data.get('length', 'medium')
        model = data.get('model', 'gpt-4')
        temperature = float(data.get('temperature', 0.7))
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        # Generate the statement
        print(f"Generating statement about: {topic}")
        result = generator.generate(
            topic=topic,
            length=length,
            temperature=temperature,
            model=model
        )
        
        if result:
            return jsonify({
                'success': True,
                'text': result,
                'topic': topic,
                'length': length,
                'model': model,
                'temperature': temperature
            })
        else:
            return jsonify({
                'error': 'Generation failed'
            }), 500
            
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/search', methods=['POST'])
@limiter.limit("20 per hour")  # Search is free, so more generous
def search():
    """Search for relevant speeches"""
    if not generator:
        return jsonify({
            'error': 'Generator not initialized. Check your API keys.'
        }), 500
    
    try:
        data = request.json
        topic = data.get('topic', '')
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        # Search for relevant speeches
        chunks = generator.search_relevant_speeches(topic, n=5)
        
        return jsonify({
            'success': True,
            'results': chunks
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Check for API keys
    if not os.environ.get('OPENAI_API_KEY'):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
    
    if not os.environ.get('PINECONE_API_KEY'):
        print("\n⚠️  WARNING: PINECONE_API_KEY not set!")
        print("Set it with: export PINECONE_API_KEY='...'")
    
    # Run the app
    print("\n" + "="*60)
    print("Starting Obama Speech Generator Web App")
    print("="*60)
    print("\nOpen your browser and go to:")
    print("http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port, debug=False)
