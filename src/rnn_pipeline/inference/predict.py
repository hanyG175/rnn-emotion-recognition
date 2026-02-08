def predict_emotion(model, tokenizer, text):
    import torch
    # 3. Make prediction
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        input_tensor = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        output = model(**input_tensor)
        probabilities = torch.softmax(output.logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()

    # 4. Map to class name
    # Based on the 'dair-ai/emotion' dataset labels:
    idx_to_label = {
        0: 'sadness',
        1: 'joy',
        2: 'love',
        3: 'anger',
        4: 'fear',
        5: 'surprise'
    }
    predicted_emotion = idx_to_label.get(predicted_class_idx, "Unknown")

    return predicted_emotion, probabilities[0][predicted_class_idx].item()