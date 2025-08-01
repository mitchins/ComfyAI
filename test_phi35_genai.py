#!/usr/bin/env python3
"""Test Phi-3.5 vision with onnxruntime_genai"""

import onnxruntime_genai as og
from pathlib import Path

def test_phi35_vision():
    try:
        # Model path
        model_path = './models/phi-3.5-vision-onnx/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4'
        
        print(f'Loading model from: {model_path}')
        config = og.Config(model_path)
        model = og.Model(config)
        processor = model.create_multimodal_processor()
        tokenizer = og.Tokenizer(model)
        print('✅ Model loaded successfully')
        
        # Load image
        pizza_path = Path('tests/pizza.jpg')
        if not pizza_path.exists():
            print(f'❌ Pizza image not found at {pizza_path}')
            return
            
        images = og.Images.open(str(pizza_path))
        print('✅ Image loaded')
        
        # Create prompt
        text = 'What is shown in this image?'
        prompt = f'<|image_1|>\n{text}'
        print(f'Prompt: {prompt}')
        
        # Process inputs
        inputs = processor(prompt, images=images)
        print('✅ Inputs processed')
        
        # Generate response
        params = og.GeneratorParams(model)
        params.set_inputs(inputs)
        params.set_search_options(max_length=50)
        
        generator = og.Generator(model, params)
        print('✅ Generator created')
        
        print('\n🍕 Generating response...')
        response = ''
        try:
            while not generator.is_done():
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                token_text = tokenizer.decode(new_token)
                response += token_text
                print(token_text, end='', flush=True)
        except Exception as e:
            print(f'\nGeneration error: {e}')
            return
        
        print(f'\n\nFinal response: {response}')
        
        # Check for food keywords
        food_keywords = ['pizza', 'food', 'cheese', 'meal', 'dish', 'slice', 'crust']
        found_food = any(keyword in response.lower() for keyword in food_keywords)
        print(f'✅ Found food keywords: {found_food}')
        
        return found_food
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_phi35_vision()
    print(f'\n{"🎉 SUCCESS!" if success else "💥 FAILED!"}')