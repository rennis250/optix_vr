# optix_vr

This is a translation of our WebGL-based VR pathtracer to OptiX, so that we can integrate
the eye tracking component of the HTC Vive Eye Pro. It is meant to be used for color perception research
and so in due time, it will have the following extra features, "imported" from the WebGL version:

- Multispectral rendering with 12-component spectra
- Multispectral texture support
- Cosine color palettes
- Elementary "foveated" rendering
- Blue noise sampling
- Fixed initial seeds per pixel to remove flickering from rendering speckle when observers look around the scene

This is a work-in-progress and somewhat proof-of-concept, as I am learning CUDA and OptiX in parallel.

This code has so far been an internal project in our lab, running mainly on my personal machine.
So far, it is only written to compile/run on Windows. Use Visual Studio to compile.

It is based on Ingo Wald's excellent intro to OptiX 7:

https://github.com/ingowald/optix7course/

I did my best to read, learn, and then write as much by myself as possible, referring back to his work
and other sources, when encountering trouble. Hence, any errors are my own.

Best wishes,
Rob
