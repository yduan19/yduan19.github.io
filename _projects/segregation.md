---
layout: page
title: Particle Segregation
description: This is part of my research in collaboration with Dow and P&G, funded by the National Science Foundation.
img: assets/img/proj_segregation/size.gif
importance: 1
category: work
related_publications: false
---



### When it comes to challenges in bulk material handling, no discussion would be complete without touching on segregation.

After water, granular media are the most ubiquitous precursor material used in industry with an estimated annual consumption exceeding 1 trillion kg.

Segregation, or "demixing," of granular materials differing in size, density, or other particle properties is problematic in many circumstances
with broad implications in situations ranging from material handling in the pharmaceutical, chemical, and processes industries to natural
phenomena such as debris flow and sediment transport.

We want to engineer approaches to suppress segregation
so that granular materials remain mixed. 

<style>
    .custom-image {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        max-width: 80%;
    }
</style>

<style>
    .custom-image1 {
        display: flex;
        justify-content: flex-end;
        align-items: right;
        max-width: 100%;
    }
</style>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/proj_segregation/size.gif" title="example image" class="img-fluid rounded z-depth-1 custom-image" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/proj_segregation/densit.gif" title="example image" class="img-fluid rounded z-depth-1 custom-image" %}
    </div>
</div>
<div class="caption">
    Separation in a rotating tumbler: rising particles remain on the periphery of the tumbler, while sinking particles reside in the core of the tumbler.<br>
    Left: Large blue and small red particles of the same density.<br>
    Right: Heavy blue and light red particles of the same size.
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


### Conference presentation at APS DFD 2020
<iframe width="560" height="315" src="https://www.youtube.com/embed/8gC2113uDBY?si=ek8OxEOSgpTqUtXR" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
