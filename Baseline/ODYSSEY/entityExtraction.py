import hydra
from tqdm import tqdm
from common.utils import *
from omegaconf import DictConfig
from model.odyssey import Odyssey
from common.prompter import Prompter


@hydra.main(config_path = "conf", config_name = "entity_extraction")
def main(cfg: DictConfig):
    with open(cfg.datasetPath, 'r') as f:
        dataset = json.load(f)

    model = Odyssey(cfg)
    prompter = Prompter(cfg)
    
    print("Entity extracting ... ")
    results = []
    for datum in tqdm(dataset):
        prompt = prompter.entityExtractionPrompt_GPT(datum)
        output = model.inference_GPT(prompt)
        post_processed_datum = prompter.processEntityExtraction_GPT(prompt, datum, output)
        results.append(post_processed_datum)

    writeJson(cfg.outputPath, flatten(results))



if __name__ == "__main__":
    main()
