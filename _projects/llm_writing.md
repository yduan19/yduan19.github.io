---
layout: page
title: llm-based paper writing
description: generate scholarly content for academic papers in the field of granular segregation
img: assets/img/llm/gemini.png
importance: 1
category: fun
related_publications: false
---


# LLM paper writing

[![Demo Page](https://img.shields.io/badge/Project-Demo-FF4B4B?logo=streamlit)](https://llmpaperwriting.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg?logo)](https://github.com/19revey/LLM_paper_writing/blob/main/LICENSE)



This AI model is designed to assist in generating scholarly content for academic papers in the area of granular segregation. It embeds queries to retrieve highly similar text chunks from relevant papers, using these documents to craft responses.

**Warning:** References might not be included in the response. Ensure all content is properly cited for publication purposes.



<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/llm/paperwritingdemo1.png"  title="example image" class="img-fluid rounded z-depth-1 custom-image" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/llm/paperwritingdemo2.png"  title="example image" class="img-fluid rounded z-depth-1 custom-image" %}
    </div>
</div>
<div class="caption">
    Left: Brief, question-style prompt for Q&A.
    <br>
    Right: Long paragraph prompt to rephrase.
</div>


## Features
- **Paragraph Generation**: Automatically generates text for academic papers based on given prompts.
- **Customization**: Ability to fine-tune the model on specific topics or styles.
- **Free to use**: Powered by the latest Gemini 1.5 pro. While there is a limit on request frequency, it is currently free.
- **Secure**: The code is open source, and uploaded PDF files are stored on Astra DB for persistence.

<img src="https://i0.wp.com/gradientflow.com/wp-content/uploads/2023/10/newsletter87-RAG-simple.png?w=1464&ssl=1" alt="Description of Image" width="500" height="300">

*Source: [Gradient Flow](https://gradientflow.com/best-practices-in-retrieval-augmented-generation/).*

## Getting Started

- The app is hosted on streamlit cloud: [https://llmpaperwriting.streamlit.app/](https://llmpaperwriting.streamlit.app/)

- To run it locally, start by configuring the necessary environmental variables:
```bash
  - GOOGLE_API_KEY
  - ASTRA_DB_ID
  - ASTRA_DB_APPLICATION_TOKEN=
```
- Next, clone this repository and launch the container:
```bash
    docker compose up
```





