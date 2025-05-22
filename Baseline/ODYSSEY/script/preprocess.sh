# 1. entity extraction
python /root/sparta/Baseline/ODYSSEY/entityExtraction.py
# 2. header extraction
python /root/sparta/Baseline/ODYSSEY/headerExtraction.py
# 3. make entity-document graph
python /root/sparta/Baseline/ODYSSEY/makeEntDocGraphs.py
# 4. entity to header mapping
python /root/sparta/Baseline/ODYSSEY/entityToHeaderMapping.py
# 5. start point initiation
python /root/sparta/Baseline/ODYSSEY/initStartPoints_faster.py