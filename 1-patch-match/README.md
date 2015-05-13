# Task: Implement the PatchMatch Algorithm

anita.sellent@tu-dresden.de
23 April 2015

You can discuss the exercises in groups, but you need to hand in individual solutions
to profit from the exercises during the examation as discussed (points give you an
indication of weight/ effort). Hand in your solution (implementation and a pdf with
a few written sentences to the questions) **before 21 May 2015**.
Sample images and the code for visualization are provided on the webpage Middlebury
benchmark webpage (http://vision.middlebury.edu/flow/).

The research paper describing the algorithm and some applications [1] is available on
the internet.

## 1. Visualization (1 Point):

Use the visualization method of the Middlebury benchmark for flow visualiza-
tion. Use some dummy flow to understand the color-coding. How can you scale
the flow to obtain an optimal visualization?

How can you scale the flow to obtain a visualization to compare different
algorithms?

## 2. The Similarity Measure I (1 Point):

Given a location in the first image, and on offset to the second image, compute
the similarity of two patches. To get started, implement the sum of squared
distances.

Find a solution what to do for patch parts that are outside the image boundary.

## 3. Implement the PatchMatch algorithm (5 Points): Use the similarity function from Exercise 2 to compute costs.

 a. Initialize the flow fields randomly.
 b. Propagte the solutions from the neighbors.
 c. Refine iteratively until the search range is smaller than 1 pixel.

Iterate between propagation and refinement. Remember to switch processing
direction between even and odd iterations. How many pixel change their match
after each iteration?

## 4. Use the image pyramid (2 Points):

Implement the pyramidal patchMatch algorithm. What is the difference in the
patchMatch pyramid approach and the pyramid approach that was introduced in
the lecture? What is the idea behind each approach?

## 5. Similarity Measure II (2 Points):

Change your similarity measure. Look at the different test-images. What are
their properties and which of the similarity measures from the lecture could be
suitable for which test-set? Implement at least one other similarity measure.

## 6. PatchMatch vs. Other Algorithms (1 Point):

There are some other optical flow methods we discussed in the lecture and also
implemented in OpenCV. Which of the algorithms would allow Exercise 5 to be
realized quickly? What do we need to do for the other algorithms?


## References

[1] Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B. Goldman. Patch-
match: A randomized correspondence algorithm for structural image editing. In
ACM Transactions on Graphics (Proc. SIGGRAPH), 2009. 2
