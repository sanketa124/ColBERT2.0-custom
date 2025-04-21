from flask import Flask, request, jsonify
from functools import lru_cache
import math
import os
from dotenv import load_dotenv
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

load_dotenv()

# Load local ASQA collection
def load_collection(file_path="data/asqa_collection.tsv"):
    with open(file_path, encoding='utf-8') as f:
        return [line.strip().split('\t', 1)[1] for line in f if line.strip()]

INDEX_NAME = 'asqa.final'  # Update this to your actual index name
INDEX_ROOT = '/home/sanket/ColBERT-Custom/ColBERT/experiments/default/indexes'      # Update this if your root dir is different
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
    probs = [math.exp(score) for score in scores]
    probs = [prob / sum(probs) for prob in probs]
    topk = []
    for pid, rank, score, prob in zip(pids, ranks, scores, probs):
        text = searcher.collection[pid]
        d = {'text': text, 'pid': pid, 'rank': rank, 'score': score, 'prob': prob}
        topk.append(d)
    topk = list(sorted(topk, key=lambda p: (-1 * p['score'], p['pid'])))
    return {"query": query, "topk": topk}

@app.route("/api/search", methods=["GET"])
def api_search():
    if request.method == "GET":
        counter["api"] += 1
        print("API request count:", counter["api"])
        return api_search_query(request.args.get("query"), request.args.get("k"))
    else:
        return ('', 405)

@app.route("/api/clarify", methods=["GET"])
def api_tree_of_clarifications():
    base_query = request.args.get("query")
    clarifications = request.args.get("clarifications", "")
    max_level = int(request.args.get("max_level", 3))
    k = int(request.args.get("k", 5))

    clarification_list = [c.strip() for c in clarifications.split(",") if c.strip()]
    levels_to_run = min(len(clarification_list), max_level)

    tree_results = []
    current_query = base_query

    for i in range(levels_to_run + 1):  # +1 to include base query
        print(f"\n[Level {i}] Query: {current_query}")
        result = api_search_query(current_query, k)
        tree_results.append({
            "level": i,
            "query": current_query,
            "results": result["topk"]
        })

        if i < levels_to_run:
            current_query += " " + clarification_list[i]

    return jsonify({
        "base_query": base_query,
        "clarifications": clarification_list[:max_level],
        "tree": tree_results
    })

if __name__ == "__main__":
    app.run("0.0.0.0", 5000)
