"""
Utility functions for generating captions
"""
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config.config import MAX_LENGTH


def word_for_id(integer, tokenizer):
    """
    Converts a token ID back to its word using reverse index mapping.
    This is O(1) lookup instead of O(n) loop.
    
    Args:
        integer: Token ID
        tokenizer: Tokenizer object with index_word attribute
        
    Returns:
        Word string or None if not found
    """
    # Use index_word if available (O(1) lookup)
    if hasattr(tokenizer, 'index_word') and tokenizer.index_word:
        return tokenizer.index_word.get(integer, None)
    
    # Fallback to word_index (O(n) lookup) - should not happen if model_loader is used
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def detect_pattern_repetition(words, pattern_length=3):
    """
    Detects if a pattern of words is repeating.
    
    Args:
        words: List of words
        pattern_length: Length of pattern to check
        
    Returns:
        True if pattern is repeating, False otherwise
    """
    if len(words) < pattern_length * 2:
        return False
    
    # Check last pattern_length words
    last_pattern = words[-pattern_length:]
    
    # Check if this pattern appears earlier in the sequence
    for i in range(len(words) - pattern_length * 2, -1, -1):
        if words[i:i+pattern_length] == last_pattern:
            return True
    
    return False


def apply_repetition_penalty(probs, recent_indices, penalty=0.5):
    """
    Applies penalty to recently used words to prevent repetition.
    
    Args:
        probs: Probability distribution
        recent_indices: List of recently used token indices
        penalty: Penalty factor (0.0 = no penalty, 1.0 = full penalty)
        
    Returns:
        Modified probability distribution
    """
    probs = probs.copy()
    for idx in recent_indices:
        if idx < len(probs):
            probs[idx] *= (1.0 - penalty)
    
    # Renormalize
    probs = probs / (np.sum(probs) + 1e-10)
    return probs


def generate_caption(model, tokenizer, image_features, max_length=None, beam_size=1, temperature=1.2, top_k=10):
    """
    Generates a caption for an image using greedy search (beam_size=1) or beam search.
    
    Args:
        model: Trained caption model (compiled)
        tokenizer: Tokenizer object with index_word attribute
        image_features: Extracted image features (shape: (1, 2048))
        max_length: Maximum caption length (defaults to config MAX_LENGTH)
        beam_size: Beam size for beam search (1 = greedy search)
        temperature: Temperature for sampling (1.0 = no change, higher = more random)
        top_k: Number of top candidates to consider
        
    Returns:
        Generated caption string
    """
    if max_length is None:
        max_length = MAX_LENGTH
    
    # Ensure index_word exists for fast lookups
    if not hasattr(tokenizer, 'index_word') or tokenizer.index_word is None:
        tokenizer.index_word = {index: word for word, index in tokenizer.word_index.items()}
    
    # Get start token index
    start_token_idx = tokenizer.word_index.get('<start>', None)
    if start_token_idx is None:
        # Try without angle brackets
        start_token_idx = tokenizer.word_index.get('start', None)
    
    # Greedy search (beam_size = 1)
    if beam_size == 1:
        # Start with just the start token index
        sequence = [start_token_idx] if start_token_idx is not None else []
        
        # Track generated words to prevent infinite loops
        generated_words = []
        recent_word_indices = []  # Track recent words for penalty
        max_recent_words = 5  # Number of recent words to penalize
        
        # Loop to generate words one by one
        for i in range(max_length):
            # Pad the sequence to the model's required length
            # Use 'post' padding to match training (zeros come after the sequence)
            padded_seq = pad_sequences([sequence], maxlen=max_length, padding='post')
            
            # Predict the next word using model() for faster inference
            yhat = model([image_features, padded_seq], training=False)
            
            # Convert to numpy array
            probs = np.array(yhat[0])
            
            # Apply repetition penalty to recently used words
            if len(recent_word_indices) > 0:
                probs = apply_repetition_penalty(probs, recent_word_indices, penalty=0.7)
            
            # Apply temperature
            if temperature != 1.0:
                probs = np.log(probs + 1e-10) / temperature
                probs = np.exp(probs - np.max(probs))
                probs = probs / np.sum(probs)
            
            # Get top-k candidates
            top_k_indices = np.argsort(probs)[-top_k:][::-1]
            
            # Try each candidate until we find a valid one
            word = None
            word_idx = None
            
            for idx in top_k_indices:
                candidate_word = word_for_id(int(idx), tokenizer)
                
                if candidate_word is None:
                    continue
                
                if candidate_word == '<end>' or candidate_word == 'end':
                    # End token found, break
                    word = candidate_word
                    break
                
                # Check if this word would create a pattern repetition
                test_words = generated_words + [candidate_word]
                if detect_pattern_repetition(test_words, pattern_length=3):
                    # Skip this word if it creates a pattern
                    continue
                
                # Check if same word repeats too many times
                if len(generated_words) >= 2 and candidate_word == generated_words[-1] == generated_words[-2]:
                    # Skip if word repeats 3 times in a row
                    continue
                
                # Valid word found
                word = candidate_word
                word_idx = int(idx)
                break
            
            # If no valid word found, break
            if word is None:
                break
            
            # Check for end token
            if word == '<end>' or word == 'end':
                break
            
            # Add word to sequence
            if word_idx is None:
                word_idx = tokenizer.word_index.get(word, None)
            
            if word_idx is not None:
                sequence.append(word_idx)
                generated_words.append(word)
                
                # Track recent words for penalty
                recent_word_indices.append(word_idx)
                if len(recent_word_indices) > max_recent_words:
                    recent_word_indices.pop(0)
                
                # Early stopping if pattern detected
                if len(generated_words) >= 6 and detect_pattern_repetition(generated_words, pattern_length=3):
                    break
            else:
                break
        
        # Convert sequence back to text
        words = []
        for idx in sequence:
            word = word_for_id(idx, tokenizer)
            if word and word != '<start>' and word != 'start':
                words.append(word)
        
        final_caption = ' '.join(words)
        
        # Final check: if caption is too repetitive, try to clean it
        if len(words) > 0:
            # Remove obvious repetitions at the end
            cleaned_words = []
            for i, word in enumerate(words):
                # Skip if same word repeats 3+ times in a row
                if i >= 2 and word == words[i-1] == words[i-2]:
                    continue
                cleaned_words.append(word)
            
            # If we removed words, use cleaned version
            if len(cleaned_words) < len(words):
                final_caption = ' '.join(cleaned_words)
        
        return final_caption
    
    else:
        # Beam search implementation (for better results)
        return generate_caption_beam_search(model, tokenizer, image_features, max_length, beam_size)


def generate_caption_beam_search(model, tokenizer, image_features, max_length, beam_size=3):
    """
    Generates a caption using beam search for better quality.
    
    Args:
        model: Trained caption model
        tokenizer: Tokenizer object
        image_features: Extracted image features
        max_length: Maximum caption length
        beam_size: Number of beams to keep
        
    Returns:
        Best generated caption string
    """
    # Ensure index_word exists
    if not hasattr(tokenizer, 'index_word') or tokenizer.index_word is None:
        tokenizer.index_word = {index: word for word, index in tokenizer.word_index.items()}
    
    # Get start token index
    start_token_idx = tokenizer.word_index.get('<start>', None)
    if start_token_idx is None:
        start_token_idx = tokenizer.word_index.get('start', None)
    
    # Initialize beam with start token index
    beams = [([start_token_idx] if start_token_idx is not None else [], 0.0, [])]  # (sequence_indices, log_probability, words)
    
    for i in range(max_length):
        candidates = []
        
        for seq_indices, log_prob, words in beams:
            # Pad sequence
            padded_seq = pad_sequences([seq_indices], maxlen=max_length, padding='post')
            
            # Predict next word probabilities
            probs = np.array(model([image_features, padded_seq], training=False)[0])
            
            # Apply repetition penalty
            if len(seq_indices) > 0:
                recent_indices = seq_indices[-5:] if len(seq_indices) >= 5 else seq_indices
                probs = apply_repetition_penalty(probs, recent_indices, penalty=0.6)
            
            # Get top candidates (more than beam_size to filter)
            top_k = beam_size * 3
            top_indices = np.argsort(probs)[-top_k:][::-1]
            
            for idx in top_indices:
                word = word_for_id(int(idx), tokenizer)
                if word is None or word == '<end>' or word == 'end':
                    continue
                
                # Check for pattern repetition
                test_words = words + [word]
                if detect_pattern_repetition(test_words, pattern_length=3):
                    continue
                
                # Check for triple repetition
                if len(words) >= 2 and word == words[-1] == words[-2]:
                    continue
                
                # Add word index to sequence
                word_idx = tokenizer.word_index.get(word, None)
                if word_idx is None:
                    continue
                
                new_seq = seq_indices + [word_idx]
                new_words = words + [word]
                new_log_prob = log_prob + np.log(probs[idx] + 1e-10)
                candidates.append((new_seq, new_log_prob, new_words))
        
        if not candidates:
            break
        
        # Keep top beam_size candidates
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # Check if all beams ended or have patterns
        all_ended = True
        for seq_indices, _, words in beams:
            if len(words) > 0:
                last_word = words[-1]
                if last_word != '<end>' and last_word != 'end':
                    all_ended = False
                    break
        if all_ended:
            break
        
        # Early stopping if patterns detected
        if len(beams) > 0 and len(beams[0][2]) >= 6:
            if all(detect_pattern_repetition(beam[2], pattern_length=3) for beam in beams):
                break
    
    # Return best sequence
    best_seq_indices, _, best_words = max(beams, key=lambda x: x[1])
    
    # Clean up words
    cleaned_words = []
    for i, word in enumerate(best_words):
        if i >= 2 and word == best_words[i-1] == best_words[i-2]:
            continue
        cleaned_words.append(word)
    
    final_caption = ' '.join(cleaned_words)
    
    return final_caption

