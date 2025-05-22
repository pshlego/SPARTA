import hydra
from tqdm import tqdm
import json
from omegaconf import DictConfig
from model.odyssey import Odyssey
from common.utils import *

@hydra.main(config_path = "conf", config_name = "inference")
def main(cfg: DictConfig):
    with open(cfg.datasetPath, "r") as f:
        dataset = json.load(f)

    model = Odyssey(cfg)

    print("Inferencing ... ")
    addEntityExtraction(dataset, cfg)
    addHeaderExtraction(dataset, cfg)
    addEntityToHeaderMapping(dataset, cfg)
    addSubtableGraph(dataset, model)
    addHybridGraph(dataset, model)
    addStartNodes(dataset, cfg)
    

    for input in tqdm(dataset):
        input['ban_list'] = []
        
        if input['start_nodes'] == []:
            model.addFinalHop(input)
        else:
            maxIdx      = input['start_scores'].index(max(input['start_scores']))
            startNodes  = [input['start_nodes'][maxIdx]]

            if startNodes[0] == "NoStartPoint":
                model.addFinalHop(input)

            else:
                G = GTLabelled()
                for n in startNodes:
                    G._get_v(n)
                input['subgraph'] = G

                if model.addOneHop(1, input):
                    if model.addOneHop(2, input):
                        if model.addOneHop(3, input):
                            model.addFinalHop(input)

        input.pop('subgraph',       None)
        input.pop('hybrid_graph',   None)
        input.pop('ed_graph',       None)
        input.pop('st_graph',       None)

        with open(cfg.outputPath, 'a+') as file:
            file.write(json.dumps(input) + '\n')


if __name__ == "__main__":
    main()
