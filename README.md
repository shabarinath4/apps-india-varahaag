**Train**  
```python main_run.py --bt_path_ids india_250620_bt --ft_path_ids india_250620_ft```  
To train base model and fine tuned model sequentially

```python main_run.py --bt_path_ids india_250620_bt```  
To train base model alone

```python main_run.py --ft_path_ids india_250620_ft```   
To train finetune model, it expects base model folder name to be specified in the yaml file  

**Inference**  
```python deploy_varaha.py --model_type base --inf_in india_250620_inf_in --inf_out india_250620_inf_out```  
To run inference, specify whether to deploy the base model or fine tuned model, specify the folder containing the data to infer on and the folder to write the output to  
