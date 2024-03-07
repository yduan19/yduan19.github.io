---
layout: page
title: llm-based resume optimizer
description: powered by Google Gemini pro
img: assets/img/resume.png
importance: 1
category: fun
related_publications: false
---



## The app is hosted on streamlit cloud: [https://llmresume.streamlit.app/](https://llmresume.streamlit.app/)

<br>
### Paste the job description and upload your resume to obtain insights, including: 
- an overall match percentage; 
- key skills that should be highlighted in your resume; 
- identification of keywords from the job description that are not present in your resume.



<br>
#### To run it locally

Make sure docker is installed, otherwise run installdocker.sh first:
```bash
sh installdocker.sh
```
Download source code:
```bash
git clone https://github.com/19revey/LLM_resume.git
```
Build docker image and start container:
```bash
docker compose up
```





### Source code is available on [github](https://github.com/19revey/LLM_resume.git)
