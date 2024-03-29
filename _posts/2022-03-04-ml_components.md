---
layout: post
title: Batch and layer normalization
date: 2022-03-4 18:20:00-0400
description: 
tags: 
categories: machine_learning
related_posts: true
---



<h2 style="color:purple;font-size: 2em;">Overview</h2>


* [Batch normalization](#section1)
* [Layer normalization](#section2)



##### Understand normalization in a neural network

  - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
  - [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
  - [https://www.pinecone.io/learn/batch-layer-normalization/](https://www.pinecone.io/learn/batch-layer-normalization/)


<a class="anchor" id="section1"></a>
<h2 style="color:purple;font-size: 2em;">Batch and layer normalization</h2>



|   |batch normalization| layer normalization|
|----|----------|---------------|
|trainable parameter   | 2*num_feature    | 2 |
|normalization across  |  num_batch   | num_feature |