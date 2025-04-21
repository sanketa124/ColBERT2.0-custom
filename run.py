from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer


def train():
    # use 4 gpus (e.g. four A100s, but you can use fewer by changing nway,accumsteps,bsize).
    with Run().context(RunConfig(nranks=1)):
        triples = 'data/asqa_triplets.json' # stripped to 1 gb for optimization
        queries = 'data/asqa_queries.tsv'
        collection = 'data/asqa_collection.tsv'

        config = ColBERTConfig(bsize=2, lr=1e-05, warmup=500, maxsteps=10000, doc_maxlen=180, dim=128, attend_to_mask_tokens=False, nway=2, accumsteps=1, similarity='cosine', use_ib_negatives=True)
        trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)

        trainer.train(checkpoint='colbert-ir/colbertv2.0')  # or start from scratch, like `bert-base-uncased`


if __name__ == '__main__':
    train()