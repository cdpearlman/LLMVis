import json
import re
import torch
from transformers import AutoTokenizer
from bertviz import head_view, model_view

# Load data and tokenizer
with open('activations.json', 'r') as f:
    data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(data['model'])

# Convert data back to tensors
input_ids = torch.tensor(data['input_ids'])

# Extract attention outputs from the new structure
attention_outputs = data['captured']['attention_outputs']

# Sort attention modules by layer number to maintain order
def extract_layer_num(module_name):
    match = re.search(r'layers\.(\d+)\.', module_name)
    return int(match.group(1)) if match else 0

sorted_attention_modules = sorted(attention_outputs.keys(), key=extract_layer_num)

# Extract attention weights (element 1) from each layer's output
# self_attn modules return (output, attention_weights) - we need the attention_weights
attentions = tuple(torch.tensor(attention_outputs[module]['output'][1]) for module in sorted_attention_modules)
raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Clean tokens: replace Ġ (BPE space marker) with proper spacing
tokens = [token.replace('Ġ', ' ') if token.startswith('Ġ') else token for token in raw_tokens]

print(f"Creating visualizations for: '{data['prompt']}'")
print(f"Cleaned tokens: {tokens}")

# Generate and save visualizations as HTML files
try:
    print("Generating head view visualization...")
    # Head view - shows attention patterns for each head
    head_html = head_view(attentions, tokens, html_action='return')
    
    # Save head view to file
    head_filename = f"bertviz/attention_head_view_{data['model'].replace('/', '_')}.html"
    with open(head_filename, 'w', encoding='utf-8') as f:
        f.write(head_html.data if hasattr(head_html, 'data') else str(head_html))
    print(f"Head view saved to: {head_filename}")
    
    print("Generating model view visualization...")
    # Model view - shows attention patterns across all layers  
    model_html = model_view(attentions, tokens, html_action='return')
    
    # Save model view to file
    model_filename = f"bertviz/attention_model_view_{data['model'].replace('/', '_')}.html"
    with open(model_filename, 'w', encoding='utf-8') as f:
        f.write(model_html.data if hasattr(model_html, 'data') else str(model_html))
    print(f"Model view saved to: {model_filename}")
    
    print("Visualizations saved successfully! Open the HTML files in your browser to view them.")
    
except Exception as e:
    print(f"Visualization error: {e}")
    print("Trying alternative approach...")
    
    # Alternative: try without html_action parameter
    try:
        from bertviz.head_view import attention_head_view
        from bertviz.model_view import attention_model_view
        
        # Try direct HTML generation
        head_html = attention_head_view(attentions, tokens)
        model_html = attention_model_view(attentions, tokens)
        
        # Save files
        with open('bertviz/attention_head_view.html', 'w', encoding='utf-8') as f:
            f.write(str(head_html))
        with open('bertviz/attention_model_view.html', 'w', encoding='utf-8') as f:
            f.write(str(model_html))
            
        print("Alternative approach successful - HTML files saved!")
        
    except Exception as e2:
        print(f"Alternative approach also failed: {e2}")
        
        # Last resort: create basic HTML files with the data
        basic_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Attention Visualization Data</title>
        </head>
        <body>
            <h1>Attention Data for: {data['prompt']}</h1>
            <h2>Tokens:</h2>
            <p>{', '.join(tokens)}</p>
            <h2>Model:</h2>
            <p>{data['model']}</p>
            <p>Attention tensor shape: {len(attentions)} layers</p>
            <p>Please check the bertviz installation and try running in Jupyter notebook for full visualization.</p>
        </body>
        </html>
        """
        
        with open('bertviz/attention_data.html', 'w', encoding='utf-8') as f:
            f.write(basic_html)
        print("Basic HTML file with attention data saved to: attention_data.html")
