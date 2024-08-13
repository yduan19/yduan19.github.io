---
layout: page
title: Fast DEM
description: GPU-enabled numerical simulation
img: assets/img/proj_dem/demtumbler.png
importance: 1
category: work
related_publications: false
---

### GPU simulation 


We want to engineer approaches to suppress segregation
so that granular materials remain mixed. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="assets/video/heapflowexperiment.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="assets/video/heapflowexperiment.mp4" class="img-fluid rounded z-depth-1" controls=true %}
    </div>
</div>
<div class="caption">
    A simple, elegant caption looks good between video rows, after each row, or doesn't have to be there at all.
</div>

It does also support embedding videos from different sources. Here are some examples:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="https://www.youtube.com/embed/jNQXAC9IVRw" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="https://player.vimeo.com/video/524933864?h=1ac4fd9fb4&title=0&byline=0&portrait=0" class="img-fluid rounded z-depth-1" %}
    </div>
</div>



### Prevent segregation by choosing the right combination of size and density

420 numerical simulations were conducted using code running on CUDA-enabled GPUs (RTX3090) for different combinations of size and density ratios to find the optimal combination of particle size ratio $$R_d$$, density ratio $$R_\rho$$, and mixture concentration $$c$$.




<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/proj_segregation/equilibrium.png" title="example image" class="img-fluid rounded z-depth-1 custom-image1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/proj_segregation/mixed.gif" title="example image" class="img-fluid rounded z-depth-1 custom-image" %}
    </div>
</div>
<div class="caption">
    Particles remain mixed along the curve for iso-concentration curves for the corresponding size  and density ratio.
</div>