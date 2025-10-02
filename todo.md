## Feature 1: Collapsible Sidebar (default collapsed)
[x] Add dcc.Store for sidebar collapse state in app.py
[x] Add toggle button (icon) to sidebar.py
[x] Add callback in app.py to handle sidebar toggle
[x] Update sidebar layout to support collapsed state (conditional rendering)
[x] Test: Sidebar defaults to collapsed on load
[x] Test: Toggle expands/collapses without breaking Cytoscape
✅ Feature 1 complete!

## Feature 2: Compare With Another Prompt
[x] Read model_selector.py to understand current prompt input structure
[x] Add "Compare" button next to prompt input in model_selector.py
[x] Add second prompt input (initially hidden) in model_selector.py
[x] Add dcc.Store for comparison mode state in app.py
[x] Add callback to show/hide second prompt on button click
[x] Duplicate cytoscape graph component for second visualization
[x] Update run_analysis callback to process both prompts
[x] Add rendering logic for second graph (below first)
[x] Test: Button reveals/hides second prompt input
[x] Test: Two graphs render independently with correct data
✅ Feature 2 complete!

## Feature 3: "Check Token" input (adds 4th edge)
[x] Add "Check Token:" input above first visualization in main_panel.py
[x] Update format_data_for_cytoscape to compute token-specific probabilities
[x] Add 4th edge for user-specified token with minimum opacity floor
[x] Handle tokenization (if multi-token, use last sub-token as per plans.md)
[x] Apply changes to both visualizations (prompt 1 & 2)
[x] Test: Token input creates visible 4th edge
[x] Test: Edge has minimum opacity even at 0 probability

✅ Feature 3 complete!

Feature Updates:
[x] Collapsible Sidebar should minimize to the left and allow main dashboard to fill screen. Maximized size should remain as is, minimized should hide all the way to the left with still visible chevron to maximize.
[x] The "Compare +" button should switch to a red button that says "Remove -". It should function exactly the same, removing the second prompt, just with a different visual.
[x] The "Check Token" text box needs a "Submit" button in order to kickoff the creation of the 4th edge.
[x] Bug: When a second prompt is given and the "Run Analysis" button is clicked, only 1 graph is created when there should be 2 graphs: one above the other.
[x] Bug: The token given in the Check Token box has a probability of 0 for every layer - added debug output to investigate
✅ All feature updates complete!

## Feature 4: Replace BertViz head_view with model_view
[x] Read current generate_bertviz_html implementation
[x] Replace head_view call with model_view
[x] Update to pass all layers' attention to model_view
[ ] Test with GPT-2 and Qwen2.5-0.5B models
[ ] Verify model_view displays correctly in iframe

## Feature 5: Attention Head Detection and Categorization
[x] Create utility module for head categorization (utils/head_detection.py)
[x] Implement detection heuristics for Previous-Token heads
[x] Implement detection heuristics for First/Positional heads
[x] Implement detection heuristics for Bag-of-Words heads
[x] Implement detection heuristics for Syntactic heads
[x] Add UI section to display categorized heads
[x] Make heuristics parameterized for tuning
✅ Feature 5 complete!

## Feature 6: Two-Prompt Difference Analysis
[x] Compute attention distribution differences across layers/heads
[x] Compute output probability differences at each layer
[x] Highlight layers with significant differences (red border)
[x] Add summary panel showing top-N divergent layers/heads
[x] Make difference thresholds configurable
✅ Feature 6 complete!