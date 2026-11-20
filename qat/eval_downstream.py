"""Downstream evaluation of base / QAT / bf16-control checkpoints.

Usage:
  python qat/eval_downstream.py --ckpt runs/s3_B_.../checkpoints/final/state.pt --precision ternary \
      --tasks mmlu cmmlu ceval gsm8k --out reports/downstream/s3_B.json
  --ckpt none evaluates the released weights (with --precision ternary: direct ternarization).

Multiple-choice tasks (MMLU test, CMMLU test, C-Eval val) are 5-shot with the subject's dev
examples; the prediction is the argmax over the next-token logits of " A".." D" after "Answer:"
("答案：" for Chinese). GSM8K test is 8-shot greedy generation (first 8 train problems as
exemplars) scored by the number after "####" / the last number in the completion.

For --precision ternary the decoder linear weights are replaced once by their dequantized ternary
values using the checkpoint's native per-row or per-group-32 scales, matching its packed 2-bit export.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qat.quant import convert_to_ternary, set_quant_group, set_quantize, ternary_dequant, ternary_modules  # noqa: E402


def quant_group_of(ckpt):
    """Read the quantization granularity recorded by the run that produced ckpt.

    Runs created before the BitCPM4 port have no key and used per-row absmean,
    so a missing value means None (legacy) rather than the new per-group default.
    """
    if not ckpt:
        return None
    path = Path(ckpt).resolve()
    for parent in path.parents:
        for name in ('resolved_config.json', 'config.json'):
            candidate = parent / name
            if candidate.is_file():
                cfg = json.loads(candidate.read_text())
                return cfg.get('quant_group')
        if parent.name == 'runs':
            break
    return None

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / 'eval_data'
CHOICES = 'ABCD'


def load(model_path, ckpt, precision, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, trust_remote_code=True)
    model.config.use_cache = True
    set_quant_group(quant_group_of(ckpt))
    convert_to_ternary(model)
    if ckpt:
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        del state
    set_quantize(model, False)
    if precision == 'ternary':
        with torch.no_grad():
            for m in ternary_modules(model):
                m.weight.copy_(ternary_dequant(m.weight.float()))
    return tok, model.to(device=device, dtype=torch.bfloat16).eval()


# ---------------------------------------------------------------- multiple choice

def read_rows(path, header):
    with open(path, encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if header:
        cols = [c.strip().lower() for c in rows[0]]
        q = cols.index('question')
        a = cols.index('answer')
        return [(r[q], r[q + 1:q + 5], r[a].strip()) for r in rows[1:]]
    return [(r[0], r[1:5], r[5].strip()) for r in rows]


def mc_task(name):
    """Yield (subject, dev_rows, eval_rows, lang) for a task."""
    if name == 'mmlu':
        d, header, lang, split = EVAL / 'mmlu' / 'data', False, 'en', 'test'
    elif name == 'cmmlu':
        d, header, lang, split = EVAL / 'cmmlu', True, 'zh', 'test'
    elif name == 'ceval':
        d, header, lang, split = EVAL / 'ceval', True, 'zh', 'val'  # test labels are not public
    mapping = json.loads((EVAL / name / 'mapping.json').read_text())
    for subject in sorted(mapping):
        if name == 'cmmlu':  # CMMLU files carry no split suffix
            dev = read_rows(d / 'dev' / f'{subject}.csv', header)
            rows = read_rows(d / split / f'{subject}.csv', header)
        else:
            dev = read_rows(d / 'dev' / f'{subject}_dev.csv', header)
            rows = read_rows(d / split / f'{subject}_{split}.csv', header)
        yield subject, mapping[subject], dev, rows, lang


def format_mc(q, opts, lang, answer=None):
    s = q.strip() + '\n' + ''.join(f'{c}. {o}\n' for c, o in zip(CHOICES, opts))
    s += 'Answer:' if lang == 'en' else '答案：'
    return s + (f' {answer}\n\n' if answer is not None else '')


def mc_prompt(subject_name, dev, q, opts, lang, shots):
    if lang == 'en':
        head = f'The following are multiple choice questions (with answers) about {subject_name}.\n\n'
    else:
        head = f'以下是关于{subject_name}的单项选择题，请直接给出正确答案的选项。\n\n'
    return head + ''.join(format_mc(dq, do, lang, da) for dq, do, da in dev[:shots]) + format_mc(q, opts, lang)


@torch.no_grad()
def run_mc(name, tok, model, device, shots, max_tokens, limit):
    choice_ids = [tok(' ' + c, add_special_tokens=False)['input_ids'][-1] for c in CHOICES]
    items = []
    for subject, info, dev, rows, lang in mc_task(name):
        for q, opts, ans in rows[:limit] if limit else rows:
            ids = tok(mc_prompt(info['name'], dev, q, opts, lang, shots))['input_ids']
            k = shots
            while len(ids) > 2048 and k > 0:  # a few C-Eval/CMMLU dev sets are very long
                k -= 1
                ids = tok(mc_prompt(info['name'], dev, q, opts, lang, k))['input_ids']
            items.append(dict(subject=subject, category=info['category'], ids=ids[-2048:], answer=ans))
    order = sorted(range(len(items)), key=lambda i: len(items[i]['ids']))
    i = 0
    while i < len(order):
        L = len(items[order[i]]['ids'])
        j = i
        while j < len(order) and (j - i + 1) * len(items[order[j]]['ids']) <= max_tokens:
            j += 1
        j = max(j, i + 1)
        batch = [items[k] for k in order[i:j]]
        width = max(len(b['ids']) for b in batch)
        # Right padding with an all-ones mask: attention is causal, so pads never affect
        # the logits at each sequence's last real position.
        x = torch.full((len(batch), width), tok.eos_token_id or 2, dtype=torch.long)
        for r, b in enumerate(batch):
            x[r, :len(b['ids'])] = torch.tensor(b['ids'])
        x = x.to(device)
        logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
        for r, b in enumerate(batch):
            b['pred'] = CHOICES[int(logits[r, len(b['ids']) - 1, choice_ids].argmax())]
        i = j
    return summarize_mc(items)


def summarize_mc(items):
    by_subj, by_cat = {}, {}
    for it in items:
        ok = it['pred'] == it['answer']
        by_subj.setdefault(it['subject'], []).append(ok)
        by_cat.setdefault(it['category'], []).append(ok)
    acc = lambda v: sum(v) / len(v)
    return {'acc': acc([it['pred'] == it['answer'] for it in items]),
            'acc_subject_avg': sum(acc(v) for v in by_subj.values()) / len(by_subj),
            'n': len(items), 'categories': {c: acc(v) for c, v in sorted(by_cat.items())}}


# ---------------------------------------------------------------- GSM8K

NUM = re.compile(r'-?[\d,]*\.?\d+')


def gsm_answer(text):
    has_marker = '####' in text
    if has_marker:
        text = text.split('####', 1)[1]
    nums = NUM.findall(text.replace('$', ''))
    if not nums:
        return None
    s = nums[0] if has_marker else nums[-1]
    s = s.replace(',', '').rstrip('.')
    try:
        return float(s)
    except ValueError:
        return None


@torch.no_grad()
def run_gsm8k(tok, model, device, shots, batch_size, max_new, limit, legacy_cache=False,
              test_file=None, fewshot_file=None):
    train = [json.loads(l) for l in Path(fewshot_file or EVAL / 'gsm8k' / 'train.jsonl').read_text().splitlines()[:shots]]
    test = [json.loads(l) for l in Path(test_file or EVAL / 'gsm8k' / 'test.jsonl').read_text().splitlines()]
    test = test[:limit] if limit else test
    head = ''.join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in train)
    prompts = [tok(head + f"Question: {e['question']}\nAnswer:")['input_ids'] for e in test]
    stop = ['\n\nQuestion:', 'Question:']
    correct, outs = 0, []
    order = sorted(range(len(test)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        width = max(len(prompts[i]) for i in idx)
        # Left padding so every sequence ends at the same position for generation.
        x = torch.full((len(idx), width), 0, dtype=torch.long)
        mask = torch.zeros_like(x)
        for r, i in enumerate(idx):
            x[r, width - len(prompts[i]):] = torch.tensor(prompts[i])
            mask[r, width - len(prompts[i]):] = 1
        cache_kwargs = {}
        if legacy_cache:
            from transformers.cache_utils import DynamicCache

            class LegacyDynamicCache(DynamicCache):
                # Official 2025 code uses these pre-Transformers-4.56 names.
                @property
                def seen_tokens(self):
                    return self.get_seq_length()

                def get_usable_length(self, new_seq_length, layer_idx=0):
                    return self.get_seq_length(layer_idx)

            cache_kwargs['past_key_values'] = LegacyDynamicCache()
        gen = model.generate(input_ids=x.to(device), attention_mask=mask.to(device), max_new_tokens=max_new,
                             do_sample=False, pad_token_id=0, eos_token_id=[2, tok.eos_token_id], **cache_kwargs)
        for r, i in enumerate(idx):
            tokens = gen[r, width:].tolist()
            eos_at = next((j for j, token in enumerate(tokens) if token in {2, tok.eos_token_id}), None)
            generated_tokens = len(tokens) if eos_at is None else eos_at + 1
            raw_text = tok.decode(tokens[:generated_tokens], skip_special_tokens=True)
            text = raw_text
            for st in stop:
                text = text.split(st)[0]
            pred, gold = gsm_answer(text), gsm_answer(test[i]['answer'])
            if gold is None:
                raise ValueError(f'No numeric gold in record {i}')
            ok = pred is not None and abs(pred - gold) < 1e-6
            correct += ok
            outs.append({'i': i, 'pair_id': test[i].get('pair_id', test[i].get('source_id', i)),
                         'source_id': test[i].get('source_id', i), 'variant_type': test[i].get('variant_type', 'original'),
                         'question': test[i]['question'], 'gold_answer': test[i]['answer'],
                         'original_answer': test[i].get('original_answer'),
                         'pred': pred, 'gold': gold, 'ok': ok, 'strict_ok': bool(ok and '####' in text),
                         'generated_tokens': generated_tokens, 'stop_reason': 'eos' if eos_at is not None else 'length',
                         'text': text, 'raw_text': raw_text})
        print(json.dumps({'event': 'gsm_progress', 'completed': min(s + batch_size, len(test)), 'total': len(test)}), flush=True)
    return {'acc': correct / len(test), 'strict_acc': sum(o['strict_ok'] for o in outs) / len(test),
            'length_stop_count': sum(o['stop_reason'] == 'length' for o in outs),
            'n': len(test), 'samples': sorted(outs, key=lambda o: o['i'])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=str(ROOT / 'MiniCPM4-0.5B-fjzcuwlf'))
    p.add_argument('--ckpt', required=True, help="state.pt / model.pt, or 'none' for the released weights")
    p.add_argument('--precision', choices=['ternary', 'float'], required=True)
    p.add_argument('--tasks', nargs='+', default=['mmlu', 'cmmlu', 'ceval', 'gsm8k'])
    p.add_argument('--out', required=True)
    p.add_argument('--mc-shots', type=int, default=5)
    p.add_argument('--mc-max-tokens', type=int, default=32768, help='tokens per MC forward batch')
    p.add_argument('--gsm-test-file', default=str(EVAL / 'gsm8k' / 'test.jsonl'))
    p.add_argument('--gsm-fewshot-file', default=str(EVAL / 'gsm8k' / 'train.jsonl'))
    p.add_argument('--legacy-cache', action='store_true', help='adapt old official DynamicCache method names')
    p.add_argument('--gsm-shots', type=int, default=8)
    p.add_argument('--gsm-batch', type=int, default=64)
    p.add_argument('--gsm-max-new', type=int, default=256)
    p.add_argument('--limit', type=int, default=0, help='per-subject (MC) / total (GSM8K) cap, for smoke tests')
    args = p.parse_args()

    device = torch.device('cuda')
    ckpt = None if args.ckpt == 'none' else args.ckpt
    tok, model = load(args.model, ckpt, args.precision, device)
    def digest(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()

    weights = [Path(ckpt)] if ckpt else sorted(Path(args.model).glob('*.safetensors'))
    if not weights:
        raise ValueError('Cannot identify model weights for evaluation provenance')
    res = {'ckpt': args.ckpt, 'precision': args.precision, 'mc_shots': args.mc_shots,
           'weight_sha256': {str(p.resolve()): digest(p) for p in weights},
           'eval_code_sha256': digest(__file__), 'prompt_version': 'question-answer-8shot-v1',
           'gsm_test_sha256': digest(args.gsm_test_file), 'gsm_fewshot_sha256': digest(args.gsm_fewshot_file),
           'model': str(Path(args.model).resolve()), 'parser_version': 2,
           'gsm_max_new': args.gsm_max_new, 'gsm_batch': args.gsm_batch, 'legacy_cache': args.legacy_cache,
           'gsm_shots': args.gsm_shots, 'limit': args.limit, 'tasks': {}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():  # keep tasks already finished by an earlier run with the same settings
        prev = json.loads(out.read_text())
        if all(prev.get(k) == v for k, v in res.items() if k != 'tasks'):
            res['tasks'] = prev['tasks']
    for task in args.tasks:
        if task in res['tasks']:
            continue
        t0 = time.time()
        if task == 'gsm8k':
            r = run_gsm8k(tok, model, device, args.gsm_shots, args.gsm_batch, args.gsm_max_new, args.limit,
                           args.legacy_cache, args.gsm_test_file, args.gsm_fewshot_file)
        else:
            r = run_mc(task, tok, model, device, args.mc_shots, args.mc_max_tokens, args.limit)
        r['seconds'] = time.time() - t0
        res['tasks'][task] = r
        print(json.dumps({task: {k: v for k, v in r.items() if k != 'samples'}}), flush=True)
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
