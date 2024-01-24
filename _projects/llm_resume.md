---
layout: page
title: ATS resume checker
description: powered by Google Gemini pro
img: assets/img/resume.png
importance: 1
category: fun
related_publications: false
---


### ATS resume checker website hosted on Amazon Web Service: 
### [http://aws-hosted-resume-checker.com](http://34.234.95.99:8501/)

<br>
#### To start

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



Paste the job description and upload your resume to obtain insights, including: 
    1. an overall match percentage; 
    2. key skills that should be highlighted in your resume; 
    3. identification of keywords from the job description that are not present in your resume.

### Source code is available on [github](https://github.com/19revey/LLM_resume.git)
