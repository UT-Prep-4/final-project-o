# Fluid Model by Lattice Boltzmann Method

## What it is

This program runs a 3D fluid model simulating the air flow around an obstacle using the Lattice Boltzmann Method with a D3Q19 lattice conf. If you don't know what this is, the Lattice Boltzmann Method works around solving the complete Navier-Stokes equations by isolating particle flow into "m" possible direction vectors within an "n" dimensional pixel (DnQm), to then calculate the probability of a cluster of air particles flowing into/out of each cell. Right now, I've got it set up calculating the air flow about a sphere, but what's cool about it is that the math is set up such that it can model the dynamics of any 3D geometry in the space.

## How to run it

- open port 6080
- connect to noVNC
- run final_project.py
- wait for the terminal to give "Launching complete aerodynamic window!"
- use the mouse to pan around!

## How it works

So, I've got the code split into 7 blocks, and we'll treat them as discrete functions, as that's essentially what they are. of them, block 5 is the most interesting, as it's where the main physics loop occurs. I use a second-order Taylor expansion of the continuous Maxwell Boltzmann distribution to model particle collisions and the flow of particle clusters.
As for debugging, the entirety of block 6 is a result of hours spent not understanding why nothing was showing up in the port because I wasn't aware that I had to convert the 32-bit numeric data that pytorch uses to crunch numbers into 64-bit data for pyVista to realize into something a little more tangible. 

## Built by
Killian Murphy with massive aid from Google Gemini.
