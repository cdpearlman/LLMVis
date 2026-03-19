-------------------------------------------------------
Qwen2ForCausalLM LOAD REPORT from: Qwen/Qwen2.5-0.5B
Key            | Status  | 
---------------+---------+-
lm_head.weight | MISSING | 

Notes:
- MISSING	:those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
-------------------------------------------------------
Traceback (most recent call last):
  File "/app/app.py", line 383, in run_generation
    results = perform_beam_search(model, tokenizer, prompt, beam_width, max_new_tokens)
  File "/app/utils/beam_search.py", line 142, in perform_beam_search
    outputs = model(seq)
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/transformers/utils/generic.py", line 843, in wrapper
    output = func(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 476, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/transformers/utils/generic.py", line 917, in wrapper
    output = func(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/transformers/utils/output_capturing.py", line 253, in wrapper
    outputs = func(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 411, in forward
    hidden_states = decoder_layer(
  File "/usr/local/lib/python3.10/site-packages/transformers/modeling_layers.py", line 93, in __call__
    return super().__call__(*args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 298, in forward
    hidden_states, _ = self.self_attn(
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 218, in forward
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 134, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
-------------------------------------------------------
Qwen2ForCausalLM LOAD REPORT from: Qwen/Qwen2.5-0.5B
Key            | Status  | 
---------------+---------+-
lm_head.weight | MISSING | 

Notes:
- MISSING	:those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
10.16.43.195 - - [19/Mar/2026 19:47:47] "POST /_dash-update-component HTTP/1.1" 200 -
10.16.43.195 - - [19/Mar/2026 19:47:55] "POST /_dash-update-component HTTP/1.1" 200 -
10.16.43.195 - - [19/Mar/2026 19:47:55] "POST /_dash-update-component HTTP/1.1" 200 -
Executing forward pass with prompt: 'Draw ascii art for a cat'
Captured 48 module outputs using PyVene
Loading weights:   0%|          | 0/290 [00:00<?, ?it/s]10.16.31.44 - - [19/Mar/2026 19:47:55] "POST /_dash-update-component HTTP/1.1" 200 -
[2026-03-19 19:47:55,972] ERROR in app: Exception on /_dash-update-component [POST]
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/site-packages/flask/app.py", line 1511, in wsgi_app
    response = self.full_dispatch_request()
  File "/usr/local/lib/python3.10/site-packages/flask/app.py", line 919, in full_dispatch_request
    rv = self.handle_user_exception(e)
  File "/usr/local/lib/python3.10/site-packages/flask/app.py", line 917, in full_dispatch_request
    rv = self.dispatch_request()
  File "/usr/local/lib/python3.10/site-packages/flask/app.py", line 902, in dispatch_request
    return self.ensure_sync(self.view_functions[rule.endpoint])(**view_args)  # type: ignore[no-any-return]
  File "/usr/local/lib/python3.10/site-packages/dash/_get_app.py", line 17, in wrap
    return ctx.run(func, self, *args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/dash/dash.py", line 1600, in dispatch
    response_data = ctx.run(partial_func)
  File "/usr/local/lib/python3.10/site-packages/dash/_callback.py", line 720, in add_context
    raise err
  File "/usr/local/lib/python3.10/site-packages/dash/_callback.py", line 711, in add_context
    output_value = _invoke_callback(func, *func_args, **func_kwargs)  # type: ignore[reportArgumentType]
  File "/usr/local/lib/python3.10/site-packages/dash/_callback.py", line 58, in _invoke_callback
    return func(*args, **kwargs)  # %% callback invoked %%
  File "/app/app.py", line 1090, in update_attribution_target_options
    options.append({'label': f"{t['token']} ({t['probability']:.1%})", 'value': t['token']})
TypeError: unsupported format string passed to NoneType.__format__
-------------------------------------------------------
DEBUG extract_layer_data: Found 24 attention modules
Loading model: gpt2-medium
Loading weights:   0%|          | 0/292 [00:00<?, ?it/s]
Loading weights:  51%|█████     | 149/292 [00:00<00:00, 1299.04it/s]
Loading weights: 100%|██████████| 292/292 [00:00<00:00, 1450.02it/s]
GPT2LMHeadModel LOAD REPORT from: gpt2-medium
Key            | Status  | 
---------------+---------+-
lm_head.weight | MISSING | 

Notes:
- MISSING	:those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
-------------------------------------------------------
10.16.43.195 - - [19/Mar/2026 20:11:03] "POST /_dash-update-component HTTP/1.1" 200 -
[2026-03-19 20:11:03,238] ERROR in app: Exception on /_dash-update-component [POST]
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/site-packages/flask/app.py", line 1511, in wsgi_app
    response = self.full_dispatch_request()
  File "/usr/local/lib/python3.10/site-packages/flask/app.py", line 919, in full_dispatch_request
    rv = self.handle_user_exception(e)
  File "/usr/local/lib/python3.10/site-packages/flask/app.py", line 917, in full_dispatch_request
    rv = self.dispatch_request()
  File "/usr/local/lib/python3.10/site-packages/flask/app.py", line 902, in dispatch_request
    return self.ensure_sync(self.view_functions[rule.endpoint])(**view_args)  # type: ignore[no-any-return]
  File "/usr/local/lib/python3.10/site-packages/dash/_get_app.py", line 17, in wrap
    return ctx.run(func, self, *args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/dash/dash.py", line 1600, in dispatch
    response_data = ctx.run(partial_func)
  File "/usr/local/lib/python3.10/site-packages/dash/_callback.py", line 720, in add_context
    raise err
  File "/usr/local/lib/python3.10/site-packages/dash/_callback.py", line 711, in add_context
    output_value = _invoke_callback(func, *func_args, **func_kwargs)  # type: ignore[reportArgumentType]
  File "/usr/local/lib/python3.10/site-packages/dash/_callback.py", line 58, in _invoke_callback
    return func(*args, **kwargs)  # %% callback invoked %%
  File "/app/app.py", line 1090, in update_attribution_target_options
    options.append({'label': f"{t['token']} ({t['probability']:.1%})", 'value': t['token']})
TypeError: unsupported format string passed to NoneType.__format__
10.16.43.195 - - [19/Mar/2026 20:11:03] "POST /_dash-update-component HTTP/1.1" 500 -
10.16.43.195 - - [19/Mar/2026 20:11:03] "POST /_dash-update-component HTTP/1.1" 200 -
Traceback (most recent call last):
  File "/app/utils/model_patterns.py", line 1337, in generate_bertviz_html
    attention_weights = torch.tensor(attention_output[1])  # [batch, heads, seq, seq]
RuntimeError: Could not infer dtype of NoneType
Traceback (most recent call last):
  File "/app/app.py", line 691, in update_pipeline_content
    outputs.append(create_output_content(
  File "/app/components/pipeline.py", line 1130, in create_output_content
    text=[f"{p:.1%}" for p in probs], textposition='outside',
  File "/app/components/pipeline.py", line 1130, in <listcomp>
    text=[f"{p:.1%}" for p in probs], textposition='outside',
TypeError: unsupported format string passed to NoneType.__format__
-------------------------------------------------------
10.16.43.195 - - [19/Mar/2026 20:26:26] "POST /_dash-update-component HTTP/1.1" 200 -
DEBUG extract_layer_data: Found 24 attention modules
Warning: Could not compute logit lens for gpt_neox.layers.0: mixed dtype (CPU): expect parameter to have scalar type of Float
Warning: Could not compute token probabilities for gpt_neox.layers.0: mixed dtype (CPU): expect parameter to have scalar type of Float
Warning: Could not compute logit lens for gpt_neox.layers.1: mixed dtype (CPU): expect parameter to have scalar type of Float
Warning: Could not compute token probabilities for gpt_neox.layers.1: mixed dtype (CPU): expect parameter to have scalar type of Float
Warning: Could not compute logit lens for gpt_neox.layers.2: mixed dtype (CPU): expect parameter to have scalar type of Float
Warning: Could not compute token probabilities for gpt_neox.layers.2: mixed dtype (CPU): expect parameter to have scalar type of Float
-------------------------------------------------------
Warning: Could not compute logit lens for gpt_neox.layers.13: mixed dtype (CPU): expect parameter to have scalar type of Float
Warning: Could not compute token probabilities for gpt_neox.layers.13: mixed dtype (CPU): expect parameter to have scalar type of Float
Warning: Could not compute logit lens for gpt_neox.layers.14: Could not infer dtype of NoneType
Warning: Could not compute token probabilities for gpt_neox.layers.14: Could not infer dtype of NoneType
Warning: Could not compute logit lens for gpt_neox.layers.15: Could not infer dtype of NoneType
Warning: Could not compute token probabilities for gpt_neox.layers.15: Could not infer dtype of NoneType
Warning: Could not compute logit lens for gpt_neox.layers.16: Could not infer dtype of NoneType
Warning: Could not compute token probabilities for gpt_neox.layers.16: Could not infer dtype of NoneType
Warning: Could not compute logit lens for gpt_neox.layers.17: Could not infer dtype of NoneType
Warning: Could not compute token probabilities for gpt_neox.layers.17: Could not infer dtype of NoneType
-------------------------------------------------------
