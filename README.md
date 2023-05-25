# captioning
caption playground



## InstructBlip:

https://github.com/salesforce/LAVIS/tree/main/projects/instructblip



### Install:

[LAVIS](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip#install-from-source):

```bash
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .
```

vicuna 13B v1.1 (huggingface format): 

```
aws s3 sync s3://mewtant-ai-models/llm/vicuna-13b-f79578f/ ./vicuna13b
```

配置vicuna:

```bash
cd ~/dev/LAVIS
nano lavis/configs/models/blip2/blip2_instruct_vicuna13b.yaml

# llm_model: "/home/ubuntu/dev/anucuiv-b31"
```

demo:

```bash
cd ~/dev/LAVIS/projects/instructblip/
python run_demo.py
```
