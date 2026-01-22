Generation Settings:
- change "Max New Tokens" label to "Number of New Tokens"
- the concept of beams is non-trivial, change "Beam" referencse to something like "Number of Generation Choices"

Generated Sequences:
- don't show the score value, it is virtually meaningless to a user

1. Tokenization
- stack tokens vertically instead of listing horizontally (more visually appealing)

2. Embedding
- add some explanation that embedding is taken from pre-learned table (improves understanding about how ID goes to embedding)

3. Attention
- use BertViz head view instead of model view, less overwhelming and lets user scroll through attention heads
- add explanation of what the user is looking at and how to navigate BertViz vizualization (answers questions "What am I looking at?", "What should this visualization show me?", "What should I look for?")
- categorize attention heads so that the different components of the model's learning are visible
- "Most attended tokens" section doesn't provide much value, remove this and focus on categorized attention heads and BertViz

4. MLP
- need more explanation about how the feed-forward networks are learned in training, allowing it to understand the current words based on its training set

5. Output
- include full prompt in "predicted next token", with the predicted token appended at the end (still highlighted)
- in top 5 tokens graph, the hover-over data should just show percent and token, not the long decimal value

Overall Conceptual Changes:
- the analysis should be done off the initial user-given prompt, not the prompt that includes max tokens
    - the selected beam will be used for comparison after experiments, not for analysis. For example, the user input is "The capital of the US is", the chosen beam is "The capital of the US is Washington D.C.", but the analysis is only done with the first prompt. After either experiment, the chosen beam is used to compare to the new results. If ablation made the output "The capital of the US is New York City", then that can be compared to the original chosen beam to show differences.
- add testing and verification to the entire project so that each round of changes can be double checked and verified for correctness (try to avoid running the app, just test functions)