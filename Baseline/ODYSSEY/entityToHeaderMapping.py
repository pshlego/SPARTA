import hydra
from tqdm import tqdm
from common.utils import *
from omegaconf import DictConfig
from model.odyssey import Odyssey
from common.prompter import Prompter

@hydra.main(config_path = "conf", config_name = "entity_to_header_mapping")
def main(cfg: DictConfig):
    with open(cfg.datasetPath, 'r') as f:
        dataset = json.load(f)

    model = Odyssey(cfg)
    prompter = Prompter(cfg)

    addEntityExtraction(dataset, cfg)
    addHeaderExtraction(dataset, cfg)

    print("Entity to Header mapping ... ")
    results = []
    for datum in tqdm(dataset):
        prompt = prompter.entityToHeaderMappingPrompt_GPT(datum)
        cnt = 0
        while cnt < 20:
            output = model.inference_GPT(prompt)
            if isinstance(output, dict):
                break
            cnt += 1
            print(f"Retry {cnt} times")
        
        if not isinstance(output, dict):
            output = {}
        
        post_processed_datum = prompter.processEntityToHeaderMapping_GPT(prompt, datum, output)
        results.append(post_processed_datum)
    
    writeJson(cfg.outputPath, flatten(results))
    

if __name__ == "__main__":
    main()
