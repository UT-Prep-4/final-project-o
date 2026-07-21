# Fluid Model by Lattice Boltzmann Method

## What it is

This program runs a fluid simulation modeling the 3D flow of air about a sphere obstacle in a wind tunnel. It uses the Lattice Boltzmann Method in D3Q19 lattice configuration -- where air flow is calculated by the probability of particle cluster movement in 19 discrete directions within a 3D pixel -- instead of solving the full Navier Stokes equations. Right now, I've got it set up calculating and modeling the dynamics of a sphere, but what's cool about it is that the math is set up such that it could model the flow about any 3D obstacle geometry.

## How to run it

- open port 6080
- connect to noVNC
- run python final_project.py in the terminal
- wait until the terminal gives "Launching complete aerodynamic window!" (may take some reloading)
- access the noVNC tab and pan with the mouse!

## How it works
REPLACE THIS: 2-4 sentences on the interesting part of the code. Which function
does the magic? What was the hardest bug you fixed?

So, I've got the code separated into 7 blocks, and we'll consider each a function, as that's pretty much what they are. Of them, block 5 plays the biggest role; it's where the main fluid simulation loop occurs. I use a second-order Taylor expansion of the continuous Maxwell-Boltzmann distribution

## Built by
Killian Murphy with massive aid from Google Gemini AI.
