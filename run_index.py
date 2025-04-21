from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="default")):

        config = ColBERTConfig(
            nbits=2,
            root="experiments/"
        )
        indexer = Indexer(checkpoint="experiments/default/none/2025-04/21/20.37.20/checkpoints/colbert", config=config)
        indexer.index(name="asqa.final", collection="data/asqa_collection.tsv")