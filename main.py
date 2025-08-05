from data_preparation import *
import yaml
import pathlib
import logging
from defragmentation import *
from QSAR_model_evaluation import *
from Similarity import *
from new_compounds import *



def load_config(path):
    """Load YAML configuration file."""
    return yaml.safe_load(open(path))

def setup_logger():
    logger = logging.getLogger("my_pipeline")
    if not logger.hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        )
    return logger

def main():
    log = setup_logger()

    config = load_config("config.yml")

    paths      = config["Paths"]
    artifacts  = config["Artifacts"]

    final_csv  = Path(paths["final_path"])
    fragments  = Path(paths["fragments"])
    rf_model   = Path(artifacts["rf_model"])
    visual_root = Path(paths["visuals_root"])
    ga_hits = Path(paths["ga_results_root"]) / config["GeneticIsland"]["top_hits_tsv"]

    if not final_csv.exists():
        log.info("⚒️  Running final_dataset.csv preparation...")
        create_final_dataset(config, log)
    else:
        log.info("final_dataset.csv already exists, skipping...")

    if not fragments.exists():
        log.info("🪚 Defragmenting dataset...")
        defragmenttion(config, log)
    else:
        log.info("fragments already exists, skipping...")

    if not rf_model.exists():
        log.info("⚙️ Building QSAR model...")
        build_qsar_model("config.yml")
    else:
        log.info("QSAR model already exists, skipping...")
    if not (visual_root / "pls_scores.csv").exists():
       log.info("📊 Building similarity visuals...")
       run_plsda_visuals(config)
    else:
      log.info("Similarity visuals already exist, skipping...")

    if not ga_hits.exists():
        log.info("🧬 Running genetic-island GA...")
        run_genetic_island(config)
    else:
        log.info("Genetic-island GA artefacts already exist, skipping...")

    log.info("🏁 Pipeline finished – 100 de-novo hits generated and ready for further optimisation.")

if __name__ == "__main__":
    main()



