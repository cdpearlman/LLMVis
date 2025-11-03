# Transformer Visualization Dashboard - Refactor Action Items

## PHASE 1: Backend Infrastructure Changes

### 1.1 Token Probability Utilities (utils/)
- [ ] Create new utility function `merge_token_probabilities()` that:
  - Takes a token string and probability dictionary
  - Finds both versions of token (with and without leading space)
  - Sums their probabilities and returns merged result
  - Example: " cat" (0.15) + "cat" (0.05) = "cat" (0.20)
- [ ] Update `extract_layer_data()` in utils to use merged token probabilities for all top-k calculations
- [ ] Update `get_check_token_probabilities()` to use merged token probabilities
- [ ] Add function `compute_global_top5_tokens()` that:
  - Takes final layer output from forward pass
  - Returns the global top 5 tokens (with merged probabilities)
  - Stores these for use across all layer visualizations

### 1.2 Layer-wise Probability Tracking
- [ ] Modify `extract_layer_data()` to track global top 5 tokens across all layers:
  - For each layer, compute probabilities of the global top 5 tokens
  - Calculate delta from previous layer (or from embedding for layer 0)
  - Store both absolute probabilities and deltas per layer
- [ ] Add function `detect_significant_probability_increases()` that:
  - Takes layer-wise probability data
  - Identifies layers where any top 5 token has ≥25% relative increase
  - Returns list of layer numbers with significant increases
  - Example: 0.20 → 0.25 is 25% increase, should be flagged

### 1.3 Attention Head Ablation
- [ ] Create new function `execute_forward_pass_with_head_ablation()` in utils that:
  - Takes model, tokenizer, prompt, config, layer_num, and head_indices
  - Zeros out specified attention head(s) in the specified layer
  - Returns activation data with ablated results
  - Maintains same data structure as current layer ablation
- [ ] Update ablation logic to support both layer-level and head-level ablation

### 1.4 Data Structure Updates
- [ ] Update activation data structure to include:
  - `global_top5_tokens`: List of 5 tokens from final layer
  - `layer_wise_top5_probs`: Dict mapping layer_num → {token: prob} for global top 5
  - `layer_wise_top5_deltas`: Dict mapping layer_num → {token: delta} for global top 5
  - `significant_layers`: List of layer numbers with ≥25% increases
- [ ] Ensure all data needed for layer expansion is computed upfront but stored efficiently

---

## PHASE 2: UI Component Removal & Cleanup ✅ COMPLETED

### 2.1 Remove Check Token Feature
- [x] Remove "Check Token" input field from `components/main_panel.py`
- [x] Remove `check-token-graph` and `check-token-graph-container` from main_panel.py
- [x] Remove `check-token-graph-store` from app.py stores
- [x] Remove `update_check_token_graph()` callback from app.py
- [x] Remove `get_check_token_probabilities()` calls from `run_analysis()` callback
- [x] Clean up any CSS related to check token feature in assets/style.css

### 2.2 Remove Bottom Experiments Section
- [x] Remove entire `experiments-section` div from `components/main_panel.py`
- [x] Remove `show_experiments_section()` callback from app.py
- [x] Remove `ablation-layer-buttons`, `ablation-selection-store`, `ablation-results-flag` stores from app.py
- [x] Remove `populate_ablation_buttons()` callback from app.py
- [x] Remove `handle_layer_selection()` callback from app.py
- [x] Remove `run_ablation_experiment()` callback from app.py (will be replaced with per-layer version)
- [x] Clean up any CSS related to experiments section in assets/style.css

### 2.3 Update Layer Accordion Display
- [x] Remove certainty display from layer accordion summary in `create_layer_accordions()` callback
- [x] Remove certainty tooltip/explanation from layer accordion content
- [x] Remove "Attention (current position)" section and top_attended display from layer accordions
- [x] Keep attention head categorization section (will be enhanced in Phase 4)

---

## PHASE 3: Tokenization Explanation Section ✅ COMPLETED

### 3.1 Create Tokenization Component
- [x] Create new file `components/tokenization_panel.py` with function `create_tokenization_panel()`
- [x] Design layout with three columns: Tokens | IDs | Embeddings
- [x] Add section title "Step 1: Tokenization & Embedding" with subtitle explaining this is the first step
- [x] For tokens column:
  - Display each token in a colored box
  - Add CSS-based colored line connecting to ID column
  - Add hover tooltip: "The input text is split into tokens, which are the basic units the model processes. Tokenization helps the model understand semantic meaning by breaking text into meaningful subwords, allowing it to handle any word, even ones it hasn't seen before."
- [x] For IDs column:
  - Display token ID numbers in boxes matching token colors
  - Add CSS-based colored line connecting to embedding column
  - Add hover tooltip: "Each token is mapped to a unique ID number from the model's vocabulary. Models have a finite vocabulary size (typically 30k-50k tokens) to keep the model size manageable while covering most language patterns through subword tokenization."
- [x] For embeddings column:
  - Display visual representation "[ ... ]" for each token
  - Add hover tooltip: "Each token ID is converted to an embedding vector - a list of numbers that represents the token's meaning. These vectors are learned during training so that words with similar meanings have similar vectors. This is the actual input fed into the transformer layers."

### 3.2 Create Static Tokenization Diagram
- [x] Create static HTML/CSS diagram showing example tokenization flow
- [x] Use placeholder text like "Hello world" → ["Hello", " world"] → [1234, 5678] → [[ ... ], [ ... ]]
- [x] Add to tokenization panel as example visualization (HTML/CSS-based, not PNG)

### 3.3 Integrate Tokenization Panel into Main Panel
- [x] Add tokenization panel to `components/main_panel.py` between prompt inputs and transformer layers section
- [x] Create callback `update_tokenization_display()` that:
  - Takes activation_data from session store
  - Extracts input_ids and uses tokenizer to decode individual tokens
  - Populates tokenization panel with actual tokens and IDs
  - Shows panel only after analysis is run
- [x] Add CSS styling for tokenization panel (colored lines, boxes, hover effects)
- [x] Ensure comparison mode shows tokenization for both prompts (stacked vertically)

---

## PHASE 4: Transformer Layers Panel Redesign

### 4.1 Create Collapsible Transformer Layers Container
- [ ] Wrap existing layer accordions in new collapsible container
- [ ] Add visual representation of stacked layers when collapsed:
  - Show layer numbers (L0, L1, L2, ..., Ln) in a stacked visual
  - Style to look like stacked panels/cards
  - Add title "Transformer Layers (Click to Expand)"
- [ ] Make entire stacked visual clickable to expand/collapse
- [ ] When collapsed, show only the stacked visual
- [ ] When expanded, show all individual layer accordions
- [ ] Add CSS animations for smooth expand/collapse transition

### 4.2 Create Top 5 Tokens By Layer Line Graph
- [ ] Create new component `create_top5_by_layer_graph()` that:
  - Takes layer_wise_top5_probs data
  - Creates Plotly line graph with:
    - X-axis: Layer numbers (0 to n)
    - Y-axis: Probability (0 to 1)
    - 5 colored lines, one per global top 5 token
    - Legend showing token names
- [ ] Highlight significant layers on graph:
  - Add vertical yellow highlighted regions for layers in significant_layers list
  - Or add yellow markers/annotations at those x-positions
- [ ] Add hover tooltip on graph explaining:
  - "This graph shows how the model's confidence in the final top 5 predictions evolves through each layer. Layers with significant probability increases (≥25% relative increase) are highlighted, indicating where the model makes important decisions. Expand the Transformer Layers panel to explore these impactful layers in detail."
- [ ] Add tooltip at bottom of graph explaining token merging:
  - "Note: Tokens with and without leading spaces (e.g., ' cat' and 'cat') are automatically merged and treated as the same token for clarity."
- [ ] Place graph below transformer layers panel in main_panel.py
- [ ] Show graph only after analysis is run
- [ ] Handle comparison mode: show two line graphs side-by-side or overlaid with different line styles

### 4.3 Add Yellow Highlighting to Significant Layer Accordions
- [ ] Modify `create_layer_accordions()` to add CSS class to significant layers
- [ ] Add CSS rule for `.layer-accordion.significant` with neon yellow border/outline
- [ ] Highlighting only visible when transformer layers panel is expanded
- [ ] Ensure highlighting works in both single and comparison modes

---

## PHASE 5: Individual Layer Panel Refactor

### 5.1 Replace Top 5 Bar Chart with Change in Token Probabilities
- [ ] Modify `create_layer_accordions()` to replace current top 5 bar chart with delta chart
- [ ] Create new function `_create_token_probability_delta_chart()` that:
  - Takes layer data with deltas for global top 5 tokens
  - Creates horizontal bar chart showing delta (change from previous layer)
  - Use green bars for positive deltas, red bars for negative deltas
  - Show token names on y-axis, delta values on x-axis
  - Add hover info showing previous prob, current prob, and delta
  - Title: "Change in Token Probabilities (from Layer N-1 to Layer N)"
- [ ] For layer 0, show change from embedding (or from 0 if no embedding probs available)
- [ ] Handle comparison mode: show grouped bars or side-by-side charts
- [ ] Place this chart as first item in layer accordion content

### 5.2 Add Visual Flow Diagram to Each Layer
- [ ] Create static diagram image showing transformer layer flow:
  - Input vector "[ ... ]" → Self-Attention → branches to:
    - Left branch: F(x) box (feed-forward network)
    - Right branch: Residual connection (curved line going around)
  - Both branches merge → Output (connects to delta chart)
- [ ] Save as `assets/layer_flow_diagram.png` or create as SVG
- [ ] Add diagram to layer accordion content after delta chart
- [ ] Add hover tooltips to diagram elements:
  - Input vector: "The output from the previous layer (or embedding layer for Layer 0) is fed as input to this layer. Each layer builds upon the representations learned by previous layers."
  - Self-Attention: "The self-attention mechanism allows each token to attend to all other tokens in the sequence, learning which tokens are most relevant for understanding context."
  - F(x) box: "Feed-forward neural networks apply learned transformations to extract meaning from the attended information and prepare representations for the next layer. These are non-linear functions that help the model learn complex patterns."
  - Residual connection: "The attention output is added back to the final output (residual connection). This preserves information from earlier layers in case the feed-forward network learns little, helping gradients flow during training and maintaining important information."
  - Output connection: "The layer's output shows how token probabilities have changed, reflecting what the layer learned."

### 5.3 Enhance Attention Head Categorization Section
- [ ] Keep existing attention head categorization display
- [ ] Add detailed tooltips for each head category:
  - Previous-Token: "These attention heads primarily focus on the immediately preceding token. They help the model track local sequential dependencies and are often important for syntax and grammar."
  - First/Positional: "These heads attend strongly to the first token or show positional patterns. They help the model maintain awareness of sentence structure and position-dependent information."
  - Bag-of-Words: "These heads distribute attention broadly across many tokens without strong positional preferences. They help aggregate semantic information from across the entire sequence."
  - Syntactic: "These heads show structured attention patterns that often correspond to syntactic relationships (e.g., subject-verb, modifier-noun). They help the model understand grammatical structure."
  - Other: "These heads show attention patterns that don't fit the above categories. They may be learning task-specific or more complex patterns."
- [ ] Add detailed BertViz usage instructions above each BertViz visualization:
  - "How to read this visualization: The left side shows Query tokens (where attention is coming FROM), and the right side shows Key tokens (where attention is going TO). Lines connect tokens that attend to each other, with thicker lines indicating stronger attention weights. Each color represents a different attention head. Double-click a color to isolate that head. Hover over lines to see exact attention weights."
- [ ] Keep BertViz visualizations within categorized head sections (current implementation)

### 5.4 Add Explore These Changes Button and Experiments
- [ ] Add "Explore These Changes" button below the flow diagram in each layer accordion
- [ ] Create collapsible experiments section that appears when button is clicked
- [ ] Add ablation experiment description:
  - Title: "Attention Head Ablation"
  - Description: "Ablation experiments remove the output of specific attention heads, exploring how the model functions without those components. We can see how important a head is by how much the predictions change when we remove it. Select one or more attention heads below to zero out their contributions and re-run the forward pass."
- [ ] Display attention head selection interface:
  - Show buttons for each attention head in the layer (e.g., "Head 0", "Head 1", ...)
  - Allow multi-select (toggle buttons)
  - Add "Run Ablation" button
- [ ] Create callback `run_head_ablation()` that:
  - Takes selected head indices and layer number
  - Calls `execute_forward_pass_with_head_ablation()`
  - Updates activation data with ablation results
  - Refreshes layer accordions to show ablated results
  - Marks ablated layer visually (similar to current layer ablation highlighting)
- [ ] Ensure ablation works in comparison mode (ablate both prompts)

---

## PHASE 6: BertViz Button Relocation

### 6.1 Move BertViz Button to Bottom
- [ ] Remove BertViz button from individual layer accordion content (currently at line 912-928 in app.py)
- [ ] Create new section below all layer accordions in main_panel.py
- [ ] Add single "View All Attention Heads Interactively (BertViz)" button
- [ ] Button should open a modal or expandable section showing full BertViz for all layers
- [ ] Update callback to show full model BertViz instead of single layer
- [ ] Keep per-category BertViz within layer accordions (no change to that)

---

## PHASE 7: CSS Styling & Polish

### 7.1 Tokenization Panel Styling
- [ ] Add CSS for tokenization panel layout (3-column grid)
- [ ] Style colored boxes for tokens, IDs, embeddings
- [ ] Create CSS-based colored connector lines between columns
- [ ] Add hover effects for tooltips
- [ ] Ensure responsive design for different screen sizes

### 7.2 Transformer Layers Panel Styling
- [ ] Style collapsed stacked layers visual (looks like stacked cards)
- [ ] Add expand/collapse animation
- [ ] Style neon yellow highlighting for significant layers (#FFFF00 or similar)
- [ ] Ensure yellow outline is visible and prominent

### 7.3 Layer Flow Diagram Styling
- [ ] Style flow diagram to be clear and readable
- [ ] Add hover effects for tooltip regions
- [ ] Ensure diagram fits well within layer accordion content

### 7.4 Top 5 Line Graph Styling
- [ ] Style line graph container
- [ ] Ensure legend is readable
- [ ] Style highlighted regions for significant layers
- [ ] Add responsive sizing

### 7.5 Experiments Section Styling
- [ ] Style "Explore These Changes" button
- [ ] Style attention head selection buttons (toggle state)
- [ ] Style ablation description text
- [ ] Ensure consistent spacing and alignment

---

## PHASE 8: Testing & Refinement

### 8.1 Single Prompt Testing
- [ ] Test full workflow with single prompt:
  - Model selection → Prompt input → Run Analysis
  - Verify tokenization panel displays correctly
  - Verify transformer layers panel collapses/expands
  - Verify top 5 line graph shows correct data
  - Verify significant layers are highlighted correctly
  - Expand individual layers and verify:
    - Delta chart shows correct changes
    - Flow diagram displays with tooltips
    - Attention head categorization works
    - BertViz loads correctly
    - Explore These Changes button works
    - Head ablation experiment works
- [ ] Test with different models (GPT-2, Qwen)
- [ ] Test with various prompt lengths

### 8.2 Comparison Mode Testing
- [ ] Test full workflow with two prompts:
  - Verify tokenization shows both prompts
  - Verify top 5 line graph handles comparison (side-by-side or overlay)
  - Verify layer accordions show comparison data
  - Verify delta charts show comparison
  - Verify ablation works for both prompts
- [ ] Test switching between single and comparison modes

### 8.3 Edge Cases & Error Handling
- [ ] Test with very short prompts (1-2 tokens)
- [ ] Test with very long prompts
- [ ] Test with special characters and unicode
- [ ] Test with prompts that produce low-confidence predictions
- [ ] Verify all tooltips display correctly
- [ ] Verify loading states and error messages

### 8.4 Performance Testing
- [ ] Verify layer data loads efficiently on expansion
- [ ] Check for memory leaks with repeated analyses
- [ ] Ensure smooth animations and transitions
- [ ] Test with slower connections (if applicable)

---

## PHASE 9: Documentation & Cleanup

### 9.1 Code Documentation
- [ ] Add docstrings to all new functions
- [ ] Update existing docstrings for modified functions
- [ ] Add inline comments for complex logic
- [ ] Document data structure changes

### 9.2 User-Facing Documentation
- [ ] Update README.md with new features
- [ ] Add screenshots of new UI
- [ ] Document workflow changes
- [ ] Add troubleshooting section if needed

### 9.3 Code Cleanup
- [ ] Remove unused imports
- [ ] Remove commented-out code
- [ ] Ensure consistent code style
- [ ] Run linter and fix any issues

### 9.4 Git Workflow
- [ ] Commit changes in logical chunks as work progresses
- [ ] Write descriptive commit messages
- [ ] Push completed phases to remote
- [ ] Create feature branches for major changes if needed

---

## Notes & Considerations

### Token Probability Merging
- Implement token merging (with/without leading space) in backend utilities
- Apply consistently across all visualizations
- Add user-facing tooltip explaining this behavior

### Significant Layer Threshold
- Currently set to 25% relative increase
- May need adjustment based on testing
- Consider making this configurable in future

### Lazy Loading vs Precomputation
- All data computed upfront during analysis
- Rendering deferred until layer expansion
- Balances performance with responsiveness

### Comparison Mode
- All features must support comparison mode
- Side-by-side or overlaid visualizations
- Consistent behavior across single and comparison modes

### Tooltip Content Quality
- Write clear, comprehensive explanations
- Assume user has basic ML knowledge but not transformer expertise
- Balance technical accuracy with accessibility

### Future Extensibility
- Design experiment framework to support future experiment types
- Keep attention head ablation modular
- Consider parameterizing thresholds and settings
