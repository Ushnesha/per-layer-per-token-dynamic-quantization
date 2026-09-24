from lm_eval import simple_evaluate

results = simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Meta-Llama-3-8B-Instruct,dtype=float16,batch_size=auto",
    tasks=["wikitext"],
)
print(results["results"])