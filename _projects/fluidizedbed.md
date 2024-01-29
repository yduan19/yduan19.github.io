---
layout: page
title: Fluidized bed
description: This is part of my PhD work, funded by Department of Energy through a grant.
img:  assets/img/proj_fluidizedbed/fluidizedcomp.png
importance: 2
category: work
related_publications: false
---

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
        max-width: 50%;
    }
</style>

### Fluidized bed has many advantages as gas-solid reactors or powder handling processors because of its advantages of high heat and mass transfer, temperature homogeneity, and mixing property. 

<br>
###### The primary challenge in simulating full-scale fluidized beds stems from significant scale differences: the apparatus spans meters, whereas the particles are mere millimeters or even smaller.

<br>
###### As it's not feasible to simulate the system by tracking each particle individually, the aim of this project is to model the particle phase as a continuous fluid. This is achieved by employing the Kinetic Theory of Granular Flow in the framework of Two Fluid Model (TFM).


<div class="row justify-content-sm-center">
    <div class="col-sm-3 mt-3  d-flex align-items-end">
        {% include figure.liquid path="assets/img/proj_fluidizedbed/fluidization" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3  d-flex align-items-end">
        {% include figure.liquid path="assets/img/proj_fluidizedbed/fluidization1" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The emergence of a bubble within a fluidized bed.
    On the left: Computational Fluid Dynamics-Discrete Element Method (CFD-DEM) simulation (5 hours to complete).
    On the right: Two Fluid Model (TFM) simulation utilizing the Kinetic Theory of Granular Flow (completed in 10 minutes).
</div>

###### Both the CFD-DEM and TFM simulations are carried out using [MFiX](https://mfix.netl.doe.gov/products/mfix/), an open-source software developed by the National Energy Technology Lab. The results are visualized using [ParaView](https://www.paraview.org/)