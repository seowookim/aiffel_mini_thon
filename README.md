# Empathetic Chatbot Project

## Introduction

This project aimed to create an empathetic chatbot.  
The dataset - which was dialogue format - can be found [here](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71305)  

We fine-tuned our dataset using KoGPT2 + Lora to classify empathy types,
and again, KoGPT2 + Lora to generate empathetic responses.  

## [EDA](EDA)

The dataset consisted of six emotional cateogries: happiness, bewilderment, anger, anxiety, sadness, and pain.  
Also, empathetic responses consisted of five categories: advice, encouragement, solace, and agreement.   
Agreement was the most prominent type in the dataset, but the distribution of empathy varied across different emotional dialogues.
![image](https://github.com/user-attachments/assets/7cffd8c8-90d9-4261-bb49-12d25b99f533)

Interestingly, advice was the most effective type of empathy that successfully changed speaker's emotion, followed by encouragment and agreement.  

## [Evaluation](evaluation)

To evaluate whether chatbot's responses are relatable, we set three evalaution standards:  

**Evaluation Criterion 1**  
Does the category of our output match the category of the original dataset's output when the same input is provided?

**Evaluation Criterion 2**  
How close is the embedding distance between the original output and our generated output? (using Voyage AI API.)

**Evaluation Criterion 3 (Human evaluation, wasn't measured for the shortage of time)**  
Did people's experiences actually shift toward a more positive direction?  
To what extent do people perceive the responses as human-like?

## [Model_training](model_training)  

We run experiment using DistillKoBERT, KoBERT, and KoGPT2. Since KoGPT2 excelled in both classification and generation tasks, we selected KoGPT2 as our final model. In addition, due to the large size of our dataset, we applied LoRA model. 

### Contributor
- Seowoo Kim: Managing, EDA, Evalaution 
- Nakyoung Kim: Model training
- Earl Lee: Model serving
- Seo Jin Lee: Model trainig, Evaluation
