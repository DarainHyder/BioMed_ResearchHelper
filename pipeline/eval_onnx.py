"""Check that the deployed (ONNX) query encoder keeps retrieval quality: held-out titles -> own paper."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ARTIFACTS, QUERY_PREFIX, WORK, is_heldout, load_papers
import onnxruntime as ort
from tokenizers import Tokenizer

def run(model_path, queries):
    tk = Tokenizer.from_file(os.path.join(ARTIFACTS, 'tokenizer.json')); tk.enable_truncation(128)
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    out = []
    for i in range(0, len(queries), 64):
        encs = tk.encode_batch([QUERY_PREFIX + q for q in queries[i:i + 64]])
        L = max(len(e.ids) for e in encs)
        ids = np.array([e.ids + [0] * (L - len(e.ids)) for e in encs], np.int64)
        m = np.array([e.attention_mask + [0] * (L - len(e.ids)) for e in encs], np.int64)
        out.append(sess.run(['embedding'], {'input_ids': ids, 'attention_mask': m, 'token_type_ids': np.zeros_like(ids)})[0])
    return np.concatenate(out)

papers = load_papers()
held = [i for i, p in enumerate(papers) if is_heldout(p['pmid'])]
E = np.load(os.path.join(ARTIFACTS, 'embeddings.f16.npy')).astype(np.float32)
for name in sys.argv[1:]:
    Q = run(name, [papers[i]['title'] for i in held])
    S = Q @ E.T
    ranks = np.array([(S[k] > S[k, i]).sum() for k, i in enumerate(held)])
    print(f'{os.path.basename(name):24s} R@1 {np.mean(ranks == 0):.3f}  R@10 {np.mean(ranks < 10):.3f}  MRR@10 {np.mean(np.where(ranks < 10, 1 / (ranks + 1), 0)):.3f}')
