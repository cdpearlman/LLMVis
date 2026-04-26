(function () {
    const STEP_KEY = 'transformer_tutorial_step';
    const DONE_KEY = 'transformer_tutorial_done';

    const FOOTER_NORMAL =
        '<br><br><i>Click <b>Continue</b> to close this window and try it yourself. ' +
        'When you are ready, click <b>Continue Tutorial</b> in the top right to move on ' +
        '— or use <b>Back</b> to revisit the previous step.</i>';

    const FOOTER_LAST =
        '<br><br><i>Click <b>Finish</b> to close the tutorial — you have seen the whole ' +
        'pipeline. <b>Back</b> still works if you want to revisit anything.</i>';

    let currentStep = 0;
    let driverObj = null;

    function safeGet(key) {
        try { return localStorage.getItem(key); } catch (e) { return null; }
    }
    function safeSet(key, value) {
        try { localStorage.setItem(key, value); } catch (e) { /* ignore */ }
    }
    function safeRemove(key) {
        try { localStorage.removeItem(key); } catch (e) { /* ignore */ }
    }

    function buildSteps() {
        const raw = [
            {
                element: '#model-dropdown',
                title: 'Step 1 of 12 — Pick a model',
                body:
                    'Everything starts here. Different transformer models have different ' +
                    'sizes, training data, and quirks, so what you see downstream depends on ' +
                    'which one you load.<br><br>' +
                    '<b>Recommended:</b> GPT-2 (124M). It is small enough to run quickly on ' +
                    'any machine and well-studied, so the attention patterns you will see ' +
                    'have known interpretations.'
            },
            {
                element: '#prompt-section',
                title: 'Step 2 of 12 — Give it something to predict',
                body:
                    'The prompt is what the model tries to continue. The four buttons above ' +
                    'load prompts chosen to activate specific, studyable behaviors.<br><br>' +
                    '<b>Recommended:</b> click <b>"Understand repetition"</b>. Its prompt ' +
                    '("The cat sat on the mat. The cat sat on the") triggers <i>induction ' +
                    'heads</i> — attention detectors that complete repeated patterns. We will ' +
                    'come back to those in the ablation step.'
            },
            {
                element: '#max-new-tokens-slider',
                title: 'Step 3 of 12 — How many words to generate',
                body:
                    'This controls how far the model extends the prompt. One word is enough ' +
                    'to see the whole pipeline in action — every extra word makes the ' +
                    'visualizations busier without adding new concepts.<br><br>' +
                    '<b>Leave this at 1</b> for your first run.'
            },
            {
                element: '#beam-width-slider',
                title: 'Step 4 of 12 — Explore alternative completions',
                body:
                    'Language models do not pick just one next word — they assign ' +
                    'probabilities to every word in the vocabulary. <i>Beam search</i> keeps ' +
                    'the top N candidates so you can compare them side-by-side.<br><br>' +
                    'Bumping this to 3 or 5 is a great way to see where the model is ' +
                    'confident vs. uncertain.'
            },
            {
                element: '#generate-btn',
                title: 'Step 5 of 12 — Run the model',
                body:
                    'Clicking Analyze runs the prompt through every layer and records the ' +
                    'internal activations. This is what powers everything below — nothing in ' +
                    'the pipeline or ablation tools is available until this runs.<br><br>' +
                    '<b>Close this window, click Analyze, and wait for the pipeline to appear</b> ' +
                    'before moving on.'
            },
            {
                element: '.stage-tokenization > summary',
                title: 'Step 6 of 12 — Stage 1: Tokenization',
                body:
                    'Models do not read words — they read <i>tokens</i>, integer IDs for ' +
                    'chunks of text. This stage shows exactly how your prompt was sliced up. ' +
                    'Surprising splits (like "running" → "run" + "ning") explain a lot of ' +
                    'later model behavior.<br><br>' +
                    'Close this window and click the stage header to expand it, then skim ' +
                    'the tokens.'
            },
            {
                element: '.stage-embedding > summary',
                title: 'Step 7 of 12 — Stage 2: Embedding',
                body:
                    'Each token ID gets converted into a vector of numbers (its "meaning") ' +
                    'plus a second vector encoding its <i>position</i> in the sentence. ' +
                    'Without position encoding, the model could not tell "dog bites man" ' +
                    'from "man bites dog".'
            },
            {
                element: '.stage-attention > summary',
                title: 'Step 8 of 12 — Stage 3: Attention',
                body:
                    'This is the heart of the transformer. Each token looks at every other ' +
                    'token and decides how much to pay attention to it. Different <i>heads</i> ' +
                    'specialize: some track previous tokens, some detect duplicates, some ' +
                    'complete patterns (induction). The bars show which specialties fired ' +
                    'most on your prompt.'
            },
            {
                element: '.stage-mlp > summary',
                title: 'Step 9 of 12 — Stage 4: Knowledge Retrieval (MLP)',
                body:
                    'After attention has shuffled information between tokens, the MLP layer ' +
                    'transforms each token vector independently. It is where the model\'s ' +
                    'factual and associative knowledge lives — expand → nonlinear activation ' +
                    '→ compress.'
            },
            {
                element: '.stage-output > summary',
                title: 'Step 10 of 12 — Stage 5: Output Selection',
                body:
                    'The final vector is projected onto the full vocabulary, producing a ' +
                    'probability for every possible next token. The chart shows the top 5 ' +
                    'candidates. This is where the abstract math becomes a concrete word.'
            },
            {
                element: '#ablation-tool',
                title: 'Step 11 of 12 — Test a head\'s importance',
                body:
                    'Now the fun part. <i>Ablation</i> means silencing a specific attention ' +
                    'head to see how much the output changes. If removing a head barely ' +
                    'affects the prediction, it was not doing much here. If it wrecks the ' +
                    'prediction, that head was critical.'
            },
            {
                element: '#ablation-category-buttons',
                title: 'Step 12 of 12 — Ablate the induction heads',
                body:
                    'Because you used the repetition prompt, the <b>Induction</b> quick-' +
                    'select button is the most interesting one to try. Close this window, ' +
                    'click Induction to add every induction head, then hit <b>Run Ablation</b>.' +
                    '<br><br>You should see the ablated output diverge from the original — ' +
                    'concrete evidence that induction heads are what make the model complete ' +
                    '"The cat sat on the" with "mat". That is mechanistic interpretability ' +
                    'in action.'
            }
        ];

        const lastIndex = raw.length - 1;
        return raw.map((s, i) => ({
            element: s.element,
            popover: {
                title: s.title,
                description: s.body + (i === lastIndex ? FOOTER_LAST : FOOTER_NORMAL)
            }
        }));
    }

    function ensureFloatBtn() {
        let btn = document.getElementById('tutorial-float-btn');
        if (btn) return btn;
        btn = document.createElement('button');
        btn.id = 'tutorial-float-btn';
        btn.className = 'tutorial-float-btn';
        btn.type = 'button';
        btn.innerHTML =
            '<span class="tutorial-float-btn-label">▶ Continue Tutorial</span>' +
            '<span class="tutorial-float-btn-dismiss" title="Dismiss tutorial">✕</span>';
        btn.addEventListener('click', (e) => {
            if (e.target.closest('.tutorial-float-btn-dismiss')) {
                e.stopPropagation();
                finishTour();
            } else {
                resumeTour();
            }
        });
        document.body.appendChild(btn);
        return btn;
    }

    function showFloatBtn() {
        const btn = ensureFloatBtn();
        btn.style.display = 'flex';
    }

    function hideFloatBtn() {
        const btn = document.getElementById('tutorial-float-btn');
        if (btn) btn.style.display = 'none';
    }

    function createDriver() {
        if (!window.driver || !window.driver.js || !window.driver.js.driver) {
            console.error('Driver.js not loaded — tutorial cannot start');
            return null;
        }
        return window.driver.js.driver({
            showProgress: false,
            allowClose: true,
            nextBtnText: 'Continue',
            doneBtnText: 'Finish',
            prevBtnText: 'Back',
            showButtons: ['previous', 'next', 'close'],
            steps: buildSteps(),
            onNextClick: () => {
                const steps = buildSteps();
                const nextIndex = currentStep + 1;
                if (nextIndex >= steps.length) {
                    if (driverObj) driverObj.destroy();
                    finishTour();
                    return;
                }
                // Step 11 (index 10) has no action for the user to take, so
                // advance straight to step 12 without minimizing to the
                // floating "Continue Tutorial" button.
                if (currentStep === 10) {
                    currentStep = nextIndex;
                    safeSet(STEP_KEY, String(currentStep));
                    driverObj.moveNext();
                    return;
                }
                if (driverObj) driverObj.destroy();
                currentStep = nextIndex;
                safeSet(STEP_KEY, String(currentStep));
                showFloatBtn();
            },
            onPrevClick: () => {
                currentStep = Math.max(0, currentStep - 1);
                safeSet(STEP_KEY, String(currentStep));
                if (driverObj) driverObj.destroy();
                showFloatBtn();
            },
            onCloseClick: () => {
                finishTour();
            }
        });
    }

    function startFresh() {
        safeRemove(DONE_KEY);
        currentStep = 0;
        safeSet(STEP_KEY, '0');
        hideFloatBtn();
        driverObj = createDriver();
        if (driverObj) driverObj.drive(0);
    }

    function resumeTour() {
        hideFloatBtn();
        driverObj = createDriver();
        if (driverObj) driverObj.drive(currentStep);
    }

    function finishTour() {
        safeSet(DONE_KEY, 'true');
        safeRemove(STEP_KEY);
        if (driverObj) {
            try { driverObj.destroy(); } catch (e) { /* ignore */ }
            driverObj = null;
        }
        hideFloatBtn();
    }

    function restoreFromStorage() {
        if (safeGet(DONE_KEY) === 'true') return;
        const saved = safeGet(STEP_KEY);
        if (saved === null) return;
        const n = parseInt(saved, 10);
        if (isNaN(n) || n < 0) return;
        currentStep = n;
        showFloatBtn();
    }

    // Event delegation — Dash renders the DOM after load, so the header button
    // may not exist when this script first runs.
    document.addEventListener('click', (e) => {
        if (e.target.closest && e.target.closest('#start-tutorial-btn')) {
            startFresh();
        }
    });

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', restoreFromStorage);
    } else {
        restoreFromStorage();
    }
})();
