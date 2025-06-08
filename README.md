# CompCausticLens
Attempt at python implementation of method for designing free form lens surfaces per Yue et al's "Poisson-Based Continuous Surface Generation for Goal-Based Caustics" & Matt Ferrarro's blog post.
Generates an STL file of the surface required to project an input image.

Uses Surf2STL write functions by asahidari/surf2stl-python 

## How to use:
- Download both .py files
- Run lens_maker(file_path) with filepath to your image of choice
- Tune parameters by fiddling with the lens_maker function, e.g the lens dimensions/focal length, number of pixels etc
