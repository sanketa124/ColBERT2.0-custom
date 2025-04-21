# colbert_lifestyle_demo.py

import colbert
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from datasets import load_dataset


def load_data(dataset='lifestyle', datasplit='dev', max_id=10000):
    print("[*] Loading dataset...")

    # Load passages
    collection_dataset = load_dataset("colbertv2/lotte_passages", dataset, trust_remote_code=True)
    collection = [x['text'] for x in collection_dataset[datasplit + '_collection']]

    # Load queries
    queries_dataset = load_dataset("colbertv2/lotte", dataset, trust_remote_code=True)
    queries = [x['query'] for x in queries_dataset['search_' + datasplit]]

    print(f"[+] Loaded {len(queries)} queries and {len(collection):,} passages")

    # Display examples
    print("\nSample Query:", queries[24])
    print("Sample Passage:", collection[19929], "\n")

    # Filter queries with relevant passage IDs < max_id
    answer_pids = [x['answers']['answer_pids'] for x in queries_dataset['search_' + datasplit]]
    filtered_queries = [q for q, apids in zip(queries, answer_pids) if any(x < max_id for x in apids)]

    print(f"[+] Filtered down to {len(filtered_queries)} queries with in-range passage IDs")

    return collection, queries, filtered_queries


def build_index(collection, index_name, checkpoint='colbert-ir/colbertv2.0', doc_maxlen=300, nbits=2, max_id=10000):
    print("[*] Starting indexing...")

    with Run().context(RunConfig(nranks=1, experiment='notebook')):
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4)
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection[:max_id], overwrite=True)

    print("[+] Indexing complete")
    print("[+] Index path:", indexer.get_index())


def run_search(index_name, collection, query, top_k=3):
    print("[*] Running search...")

    with Run().context(RunConfig(experiment='notebook')):
        searcher = Searcher(index=index_name, collection=collection)

        print(f"\n#> Query: {query}")
        results = searcher.search(query, k=top_k)

        print("\nTop-k Results:")
        for passage_id, passage_rank, passage_score in zip(*results):
            print(f"\t [{passage_rank}] \t Score: {passage_score:.1f} \t Passage: {searcher.collection[passage_id]}")


if __name__ == "__main__":
    DATASET = 'lifestyle'
    SPLIT = 'dev'
    MAX_DOCS = 10000
    NBITS = 2
    INDEX_NAME = f'{DATASET}.{SPLIT}.{NBITS}bits'

    collection, queries, filtered_queries = load_data(dataset=DATASET, datasplit=SPLIT, max_id=MAX_DOCS)
    build_index(collection, index_name=INDEX_NAME, max_id=MAX_DOCS, nbits=NBITS)

    # Example search
    example_query = filtered_queries[13]
    run_search(INDEX_NAME, collection, query=example_query, top_k=3)
