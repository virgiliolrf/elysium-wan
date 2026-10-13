import os, sys  # added on commit (README.md)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")).read())
# ADR-0009 round 1, E2 control "thread": a Model.forward that threads a non-residual value from layer to layer.
# Expected: proto2's NaN stubs make the gate DIFFER (caught); replay stubs traced on the gate's own ids pass (blind);
# replay stubs traced on independent ids DIFFER.
import os, sys
sys.argv = [sys.argv[0]]
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "proto2.py")).read().split("if __name__ != '__main__'")[0]
exec(SRC)
import transformers.models.llama.modeling_llama as ML
from transformers.modeling_outputs import BaseModelOutputWithPast
orig_layer = ML.LlamaDecoderLayer.forward
def layer_fwd(self, hidden_states, thread=None, **kw):
    h = orig_layer(self, hidden_states, **kw)
    if thread is not None: h = h + 0.125 * thread
    return (h, h * 0.5)
def model_fwd(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
    inputs_embeds = self.embed_tokens(input_ids)
    position_ids = torch.arange(inputs_embeds.shape[1]).unsqueeze(0)
    mask = ML.create_causal_mask(config=self.config, inputs_embeds=inputs_embeds, attention_mask=None, past_key_values=None, position_ids=position_ids)
    h = inputs_embeds; pe = self.rotary_emb(h, position_ids=position_ids); thread = None
    for layer in self.layers[: self.config.num_hidden_layers]:
        h, thread = layer(h, thread=thread, attention_mask=mask, position_embeddings=pe, position_ids=position_ids)
    return BaseModelOutputWithPast(last_hidden_state=self.norm(h))
ML.LlamaDecoderLayer.forward = layer_fwd; ML.LlamaModel.forward = model_fwd
for ctrl, same in (("", ""), ("replay", "1"), ("replay", "")):
    CTRL = ctrl; globals()["CTRL"] = ctrl
    if same: os.environ["SAMEIDS"] = "1"
    else: os.environ.pop("SAMEIDS", None)
    tag = f"thread stubs={'replay' if ctrl else 'nan'} trace_ids={'gate' if same else 'independent'}"
    try: r = check("llama", dict(BASE), torch.float32, M=1)
    except Exception as e: print(f"{tag}: refused ({type(e).__name__}: {str(e)[:90]})", flush=True); continue
    print(f"{tag}: logits_equal={r['logits']} grads_differ={r['differ']}/{r['grads']}", flush=True)
