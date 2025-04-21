from flask import Flask, request
from functools import lru_cache
import math
import os
from dotenv import load_dotenv

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

load_dotenv()

# Load passages from the ASQA TSV file
def load_collection(path='data/asqa_collection.tsv', max_id=None):
    with open(path, encoding='utf-8') as f:
        collection = [line.strip().split('\t', 1)[1] for line in f]
        return collection[:max_id] if max_id else collection

# Update these to your index name and index root
INDEX_NAME = 'asqa.final'  # match the name used during indexing
INDEX_ROOT = '/home/sanket/ColBERT-Custom/ColBERT/experiments/default/indexes'  # match your ColBERT indexing output directory
COLLECTION = load_collection()

app = Flask(__name__)

searcher = Searcher(index=INDEX_NAME, index_root=INDEX_ROOT, collection=COLLECTION)
counter = {"api": 0}

@lru_cache(maxsize=1000000)
def api_search_query(query, k):
    print(f"Query={query}")
    if k is None:
        k = 10
    k = min(int(k), 100)

    pids, ranks, scores = searcher.search(query, k=100)
    pids, ranks, scores = pids[:k], ranks[:k], scores[:k]
    passages = [searcher.collection[pid] for pid in pids]
    probs = [math.exp(score) for score in scores]
    probs = [prob / sum(probs) for prob in probs]

    topk = []
    for pid, rank, score, prob in zip(pids, ranks, scores, probs):
        text = searcher.collection[pid]
        d = {'text': text, 'pid': pid, 'rank': rank, 'score': score, 'prob': prob}
        topk.append(d)

    topk = sorted(topk, key=lambda p: (-p['score'], p['pid']))
    return {"query": query, "topk": topk}

@app.route("/api/search", methods=["GET"])
def api_search():
    if request.method == "GET":
        counter["api"] += 1
        print("API request count:", counter["api"])
        return api_search_query(request.args.get("query"), request.args.get("k"))
    else:
        return ('', 405)

if __name__ == "__main__":
    app.run("0.0.0.0", 5000)
