"""
Obama Speech Generator using Pinecone Vector Database
This version uses semantic search instead of keyword matching
"""

import os
from typing import List, Dict

class ObamaRAGGeneratorPinecone:
    def __init__(self, openai_api_key: str = None, pinecone_api_key: str = None, 
                 index_name: str = "obama-speeches"):
        """
        Initialize the generator with Pinecone
        
        Args:
            openai_api_key: OpenAI API key
            pinecone_api_key: Pinecone API key
            index_name: Name of the Pinecone index
        """
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        self.pinecone_api_key = pinecone_api_key or os.environ.get('PINECONE_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY environment variable.")
        
        # Connect to Pinecone
        print("Connecting to Pinecone...")
        try:
            from pinecone import Pinecone
            import openai
            
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self.index = self.pc.Index(index_name)
            self.openai_client = openai
            openai.api_key = self.openai_api_key
            
            # Check connection
            stats = self.index.describe_index_stats()
            print(f"✓ Connected to Pinecone index: {index_name}")
            print(f"  Total vectors: {stats['total_vector_count']}")
            
        except ImportError as e:
            print(f"ERROR: Missing package - {e}")
            print("Run: pip install pinecone-client openai")
            raise
        except Exception as e:
            print(f"ERROR: Could not connect to Pinecone - {e}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """Create embedding for search query"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        return response.data[0].embedding
    
    def search_relevant_speeches(self, topic: str, n: int = 3) -> List[Dict]:
        """
        Search for relevant speech chunks using semantic similarity
        
        Args:
            topic: What to search for
            n: Number of results to return
        
        Returns:
            List of relevant speech chunks with metadata
        """
        # Create embedding for the topic
        query_embedding = self.embed_query(topic)
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=n,
            include_metadata=True
        )
        
        # Extract matches
        chunks = []
        for match in results['matches']:
            chunks.append({
                'text': match['metadata']['text'],
                'title': match['metadata']['title'],
                'date': match['metadata']['date'],
                'url': match['metadata']['url'],
                'score': match['score']
            })
        
        return chunks
    
    def create_prompt(self, topic: str, n_examples: int = 3, length: str = "medium") -> str:
        """
        Create a prompt with relevant speech examples from Pinecone
        
        Args:
            topic: What to generate a statement about
            n_examples: Number of example speeches to include
            length: "short" (1 paragraph), "medium" (2-3 paragraphs), "long" (5+ paragraphs)
        """
        relevant_chunks = self.search_relevant_speeches(topic, n=n_examples)
        
        length_instructions = {
            "short": "Write 1 substantial paragraph (4-6 sentences).",
            "medium": "Write 2-3 paragraphs.",
            "long": "Write 5-7 paragraphs with a clear beginning, middle, and end."
        }
        
        prompt = f"""You are tasked with generating a statement in Barack Obama's authentic speaking style.

Here are {len(relevant_chunks)} relevant examples of how Barack Obama speaks (retrieved by semantic similarity):

"""
        
        for i, chunk in enumerate(relevant_chunks, 1):
            prompt += f"""
--- Example {i}: "{chunk['title']}" ({chunk['date']}) [Relevance: {chunk['score']:.2f}] ---
{chunk['text']}

"""
        
        prompt += f"""
--- End of Examples ---

Now, generate a statement in Barack Obama's voice about: "{topic}"

Key characteristics of Obama's speaking style to emulate:
• Thoughtful and measured tone, acknowledging complexity
• Personal stories and connections to everyday Americans
• Balanced perspective - hopeful while acknowledging real challenges
• Appeals to shared values and common ground
• Clear, accessible language with occasional rhetorical flourishes
• Use of "we" and "our" to build unity
• Specific examples and concrete details
• Inspirational closing that calls people to action

{length_instructions.get(length, length_instructions["medium"])}

Statement:"""
        
        return prompt
    
    def generate(self, topic: str, length: str = "medium", temperature: float = 0.7,
                model: str = "gpt-4") -> str:
        """
        Generate an Obama-style statement on the given topic
        
        Args:
            topic: What to write about
            length: "short", "medium", or "long"
            temperature: 0.0-1.0, higher = more creative
            model: "gpt-4" or "gpt-3.5-turbo" (cheaper)
        """
        prompt = self.create_prompt(topic, n_examples=3, length=length)
        
        print(f"\n🔍 Searching for relevant speeches about: {topic}")
        print(f"📝 Length: {length}, Temperature: {temperature}, Model: {model}")
        print("🤖 Generating...")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=1000 if length == "long" else 500
            )
            
            generated_text = response.choices[0].message.content
            
            print("\n" + "="*60)
            print("GENERATED STATEMENT")
            print("="*60)
            print(generated_text)
            print("="*60)
            
            return generated_text
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return None
    
    def interactive_mode(self):
        """Run in interactive mode - keep generating statements"""
        print("\n" + "="*60)
        print("OBAMA SPEECH GENERATOR - Interactive Mode (Pinecone)")
        print("="*60)
        print("\nCommands:")
        print("  - Enter a topic to generate a statement")
        print("  - Type 'short', 'medium', or 'long' to change length")
        print("  - Type 'gpt4' or 'gpt3' to switch models")
        print("  - Type 'search <topic>' to preview relevant speeches")
        print("  - Type 'quit' to exit")
        print("="*60 + "\n")
        
        length = "medium"
        model = "gpt-4"
        
        while True:
            user_input = input("\nTopic (or command): ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower() in ['short', 'medium', 'long']:
                length = user_input.lower()
                print(f"✓ Length set to: {length}")
                continue
            
            if user_input.lower() == 'gpt4':
                model = "gpt-4"
                print(f"✓ Model set to: {model}")
                continue
            
            if user_input.lower() == 'gpt3':
                model = "gpt-3.5-turbo"
                print(f"✓ Model set to: {model} (cheaper)")
                continue
            
            if user_input.lower().startswith('search '):
                topic = user_input[7:].strip()
                print(f"\n🔍 Searching for speeches about: {topic}")
                chunks = self.search_relevant_speeches(topic, n=5)
                print(f"\nFound {len(chunks)} relevant chunks:\n")
                for i, chunk in enumerate(chunks, 1):
                    print(f"{i}. {chunk['title']} ({chunk['date']})")
                    print(f"   Relevance: {chunk['score']:.2f}")
                    print(f"   Preview: {chunk['text'][:150]}...")
                    print()
                continue
            
            # Generate
            result = self.generate(user_input, length=length, model=model)
            
            if result:
                # Ask if they want to save
                save = input("\n💾 Save this statement? (y/n): ").strip().lower()
                if save == 'y':
                    filename = f"generated_{user_input.replace(' ', '_')[:30]}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"Topic: {user_input}\n")
                        f.write(f"Length: {length}\n")
                        f.write(f"Model: {model}\n")
                        f.write("="*60 + "\n\n")
                        f.write(result)
                    print(f"✓ Saved to: {filename}")


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("OBAMA SPEECH GENERATOR (Pinecone Version)")
    print("="*60)
    
    # Check for API keys
    openai_key = os.environ.get('OPENAI_API_KEY')
    pinecone_key = os.environ.get('PINECONE_API_KEY')
    
    if not openai_key:
        print("\n❌ ERROR: OPENAI_API_KEY not set")
        print("Run: export OPENAI_API_KEY='sk-...'")
        exit(1)
    
    if not pinecone_key:
        print("\n❌ ERROR: PINECONE_API_KEY not set")
        print("Run: export PINECONE_API_KEY='your-key-here'")
        exit(1)
    
    try:
        # Initialize generator
        generator = ObamaRAGGeneratorPinecone()
        
        # Run in interactive mode
        generator.interactive_mode()
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nMake sure you've:")
        print("1. Run upload_to_pinecone.py first")
        print("2. Set both OPENAI_API_KEY and PINECONE_API_KEY")
        print("3. Installed required packages: pip install pinecone-client openai")
