# The implementation code of "Leveraging Prior Experience: An Expandable Auxiliary Knowledge Base for Text-to-SQL"

## Download BIRD dataset
Download [BIRD data](https://drive.google.com/drive/folders/1zcoVq3SZItFaTIc6HA7AR_eMdZqKVjpL?usp=sharing) and 保存到/LPE-sql/data/


## Run Inference
Add your openai key in the *gpt_request.py*. 
```shell
api_key="your_api_key"",
base_url="your_base_url"",
```

Run the command below, and the predicted sql will be save to the file named "predict_dev.json" 位于/LPE-sql/result/engine (Llama-3.1-70B)。 the predicted result will be save to the file named 'result.txt' 位于/LPE-sql/src/knowledge_base/results/engine(Llama-3.1-70B)
```bash
sh run.sh
```

## Run evaluation
```bash
python evaluation_ex.py --path /LPE-sql/src/knowledge_base/results/engine/result.txt 
```
