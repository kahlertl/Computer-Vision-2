# Exercise 2: Image segmentation - Experiments with GrabCut

04 June 2015

## Conditions

You can discuss the exercises in groups, but you need to hand in individual solutions to profit from
the exercises during the examination as discussed (points give you an indication of weight / effort).
Hand in your solution (implementation and a few written sentences to the questions) until 25 June
2015 to Holger.Heidrich @ TU Dresden.


## 1. GrabCut example (2P)

Run the [OpenCV GrabCut example](http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html?highlight=grabcut#grabcut).
Understand it from debugging, the lectures and the [paper](http://wwwpub.zih.tu-dresden.de/~cvweb/publications/papers/2004/siggraph04-grabcut.pdf).

### Images

 - [fruits.jpg](https://github.com/Itseez/opencv/blob/master/samples/data/fruits.jpg)
 - [lama.bmp](http://cvlab-dresden.de/wp-content/uploads/2015/06/lama.bmp)
 - [4 faces](http://cvlab-dresden.de/wp-content/uploads/2015/06/face1-4.png)


## 2. Modifing GrubCut

Change the code (by grabbing the relevant source from opencv):


### 2.1 Smarter forground selection (4P)

Instead of the whole inner rectangle use only those pixels for foreground
distribution that are most unlikely in the background distribution. The portion
of that pixels shall be a slider parameter.


### 2.2 Extended model for pairwise termins (3P)

```latex
E : {0, 1}^n \in R
E(x) = ∑ θ i ( x i , z i ) + i ∑ i , j ∈ N θ i , j ( x i , x j , z i , z j )
```

where `x` is the unknown segmentation, and `z` the given image. Also, the
neighbourhood system `N` can be 4- or 8-connected.

`θ_i(x_i , z_i)` is the colour likelihood, computed via Gaussian Mixture models or histograms.

```latex
θ_{i,j} (x_i, x_j, z_i, z_j) = |x_i - x_j| [ λ 1 +λ 2 (−exp { −β ||z i −z j ∥ 2 }) ]
β=2(Mean (∥z i −z j ∥)) −1
```

this means `λ_1` represents the standard Ising prior, and `λ_2` the contrast sensitive term.
Use a meaningful slider for both.

Understand what the different parameters are doing by choosing extreme values
and different images:

 1. With only `λ_1` switched on, analyse the difference between an only 4-connected and 8-
    connected neighbourhood system. What do you see?

 2. Switch on the contrast (edge) sensitive information `λ_2`. Vary the relative weights of λ 1
    and `λ_2` . What do you see? Does the effect make sense?
