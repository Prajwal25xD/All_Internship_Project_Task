# Repetition Issue Fix

## 🔍 Problem Identified

The model was generating the same word repeatedly (e.g., "stroke stroke stroke..."). This was caused by:

1. **Incorrect Sequence Building**: Using `texts_to_sequences()` on the entire string each time, which could cause issues with how the sequence was being processed
2. **No Repetition Detection**: The model could get stuck in a loop without any mechanism to break out
3. **Sequence Handling**: The way sequences were being built and fed to the model wasn't optimal

## ✅ Solution Implemented

### 1. **Direct Token Index Management**
- Changed from building text strings to working directly with token indices
- More reliable and matches how the model was trained
- Prevents issues with tokenization of growing strings

### 2. **Repetition Detection and Prevention**
- Added tracking of consecutive word repeats
- If the same word repeats 3+ times, the system:
  - Tries to find an alternative word from top-k candidates
  - Breaks the loop if no alternative is found
  - Prevents infinite repetition loops

### 3. **Improved Sequence Building**
- Start with start token index directly
- Build sequence by appending token indices
- Convert to words only at the end
- More efficient and reliable

### 4. **Better End Token Handling**
- Checks for both '<end>' and 'end' tokens
- Properly handles end of sequence

### 5. **Top-K Sampling Fallback**
- When repetition is detected, samples from top-5 candidates
- Helps break out of repetitive patterns
- Provides alternative words when stuck

## 📝 Code Changes

### Before (Problematic):
```python
in_text = '<start>'
for i in range(max_length):
    sequence = tokenizer.texts_to_sequences([in_text])[0]  # Problem: tokenizing entire string each time
    sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
    yhat = model([image_features, sequence], training=False)
    word = word_for_id(np.argmax(yhat[0]), tokenizer)
    in_text += ' ' + word  # Problem: keeps appending, might cause issues
```

### After (Fixed):
```python
start_token_idx = tokenizer.word_index.get('<start>', None)
sequence = [start_token_idx] if start_token_idx is not None else []
generated_words = []
consecutive_repeats = 0

for i in range(max_length):
    padded_seq = pad_sequences([sequence], maxlen=max_length, padding='post')
    yhat = model([image_features, padded_seq], training=False)
    word = word_for_id(np.argmax(yhat[0]), tokenizer)
    
    # Repetition detection
    if len(generated_words) > 0 and word == generated_words[-1]:
        consecutive_repeats += 1
        if consecutive_repeats >= 3:
            # Try top-k sampling to break repetition
            # ... fallback logic ...
    
    word_idx = tokenizer.word_index.get(word, None)
    if word_idx is not None:
        sequence.append(word_idx)  # Build with indices
        generated_words.append(word)
```

## 🎯 Key Improvements

1. **Direct Index Management**: Working with token indices instead of text strings
2. **Repetition Prevention**: Automatic detection and handling of repetitive patterns
3. **Top-K Fallback**: Alternative word selection when stuck
4. **Better Sequence Building**: More reliable sequence construction
5. **Improved Error Handling**: Better handling of edge cases

## 🧪 Testing

To test the fix:

1. **Run the app**:
   ```bash
   streamlit run app.py
   ```

2. **Upload an image** and generate a caption

3. **Check for**:
   - No repeated words (or minimal repetition)
   - Proper caption generation
   - No infinite loops

## 🔧 Configuration

The repetition detection can be adjusted in `utils/caption_generator.py`:

```python
max_consecutive_repeats = 3  # Maximum allowed consecutive repeats
top_k = 5  # Number of top candidates to consider when breaking repetition
```

## 📊 Expected Results

- **Before**: "stroke stroke stroke stroke..." (infinite repetition)
- **After**: Proper captions like "a dog running on the beach" or "a man sitting on a bench"

## 🐛 Troubleshooting

If repetition still occurs:

1. **Check tokenizer**: Ensure tokenizer is properly loaded
2. **Verify model weights**: Ensure model weights match the architecture
3. **Try beam search**: Beam search often produces better results
4. **Adjust parameters**: Increase `max_consecutive_repeats` or `top_k` if needed

## 📝 Notes

- The fix works with both greedy search and beam search
- Beam search is recommended for better quality captions
- The repetition detection is conservative (allows up to 3 repeats) to avoid false positives

---

**Last Updated**: After fixing repetition issue
**Version**: 2.1 (Repetition Fix)

