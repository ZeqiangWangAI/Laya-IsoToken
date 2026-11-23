import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

from .codebook import Code, write_codebook
from .models import NativeDecision


def tiny_model(output):
    output.mkdir(parents=True, exist_ok=False)
    words = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", "asthma", "diabetes", "infection", "oxygen",
             "patient", "has", "no", "yes", "the", "code", "is", "supported", "choice", "question",
             "Does", "this", "clinical", "note", "support", "assigning", "Any", "of", "these", "conditions",
             "applies", "OR", "icd9_diag", "icd9_proc", "A", "B", ":", "?", "(", ")", ",", "not"]
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, cls_token="[CLS]", sep_token="[SEP]",
                                  mask_token="[MASK]", pad_token="[PAD]", unk_token="[UNK]")
    fast.save_pretrained(output / "tokenizer")
    config = ModernBertConfig(vocab_size=len(words), hidden_size=64, intermediate_size=128,
                             num_hidden_layers=2, num_attention_heads=1, max_position_embeddings=512,
                             local_attention=64, global_attn_every_n_layers=2, pad_token_id=0,
                             cls_token_id=1, sep_token_id=2, bos_token_id=1, eos_token_id=2)
    config.reference_compile = False
    config.save_pretrained(output / "encoder")
    torch.manual_seed(7)
    encoder = ModernBertModel(config)
    native = NativeDecision(encoder, head_layers=1)
    save_file(native.state_dict(), output / "model.safetensors")
    (output / "rl_agent_config.json").write_text(json.dumps({"head_layers": 1}))
    (output / "origin.json").write_text(json.dumps({"repo": "synthetic", "revision": "fixture-v1", "synthetic": True}))


def fixture(output, model_dir=None):
    from .cli import parser
    from .data import prepare
    output.mkdir(parents=True, exist_ok=False)
    if model_dir is None:
        model_dir = output / "model"
        tiny_model(model_dir)
    codes = [Code("icd9_diag:SYN_A", "icd9_diag", "SYN_A", "asthma"),
             Code("icd9_diag:SYN_D", "icd9_diag", "SYN_D", "diabetes"),
             Code("icd9_diag:SYN_I", "icd9_diag", "SYN_I", "infection"),
             Code("icd9_proc:SYN_O", "icd9_proc", "SYN_O", "oxygen")]
    book = output / "codes.json"
    write_codebook(codes, book)
    for split, count in (("train", 8), ("validation", 4)):
        records = [synthetic_row(split, i) for i in range(count)]
        raw = output / f"{split}.jsonl"
        raw.write_text("".join(json.dumps(row) + "\n" for row in records))
        args = parser().parse_args(["prepare", "--input", str(raw), "--codebook", str(book),
                                   "--model-dir", str(model_dir), "--output", str(output / split),
                                   "--split", split, "--dataset-version", "synthetic-v1", "--chunk-size", "64", "--max-chunks", "2"])
        prepare(args)
    return Path(model_dir)


def synthetic_row(split, index):
    disease = [("SYN_A", "asthma"), ("SYN_D", "diabetes"), ("SYN_I", "infection")][index % 3]
    procedure = index % 2 == 0
    return {"_id": f"{split}-{index}", "subject_id": f"{split}-patient-{index // 2}",
            "text": "patient has " + disease[1] + (" oxygen" if procedure else ""),
            "icd9_diag": [disease[0]], "icd9_proc": ["SYN_O"] if procedure else []}


def smoke(args):
    from .cli import parser
    from .train import train
    from .evaluate import evaluate
    torch.set_num_threads(2)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model_dir = fixture(output / "fixture", Path(args.model_dir) if args.model_dir else None)
    summaries = {}
    base = ["--train-data", str(output / "fixture/train"), "--validation-data", str(output / "fixture/validation"),
            "--model-dir", str(model_dir), "--device", args.device, "--epochs", "1", "--batch-size", "1",
            "--candidates", "4", "--merges", "2", "--projection-dim", "32", "--warmup-steps", "0",
            "--validation-limit", "2"]
    for kind in ("native", "evidence", "evidence_plain", "duallaat", "linear"):
        command = ["train", *base, "--kind", kind, "--max-steps", "1", "--output", str(output / kind)]
        summaries[kind] = train(parser().parse_args(command))
    checkpoint = output / "evidence/step-00000001.pt"
    resumed = parser().parse_args(["train", *base, "--kind", "evidence", "--max-steps", "2",
                                   "--resume", str(checkpoint), "--output", str(output / "resumed")])
    summaries["resumed"] = train(resumed)
    command = ["evaluate", "--checkpoint", str(output / "resumed/step-00000002.pt"),
               "--model-dir", str(model_dir), "--data", str(output / "fixture/validation"),
               "--device", args.device, "--output", str(output / "evaluation")]
    evaluate(parser().parse_args(command))
    (output / "smoke.json").write_text(json.dumps({"data": "synthetic", "runs": summaries}, indent=2))
    print(f"Smoke checks complete: {output}")

