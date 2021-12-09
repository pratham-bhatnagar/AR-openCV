<div align="center">
  <h1> Augmented Reality using openCV-python</h1>
  <img alt="output" src="assets/output.gif" />
 </div>

> This project uses openCV for showcasing the use if openCV-pyhton in Augmented Reality.


# 💾 REQUIREMENTS
+ opencv-python
+ numpy

```bash
pip install -r requirements.txt
```

### ORB in OpenCV
ORB (oriented BRIEF) keypoint detector and descriptor extractor.

The algorithm uses FAST in pyramids to detect stable keypoints, selects the strongest features using FAST or Harris response, finds their orientation using first-order moments and computes the descriptors using BRIEF (where the coordinates of random point pairs (or k-tuples) are rotated according to the measured orientation).

> Source: [ORB openCV](https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html)

### Brute-Force Feature Matching
Brute-Force matcher is simple. It takes the descriptor of one feature in first set and is matched with all other features in second set using some distance calculation. And the closest one is returned.

For BF matcher, first we have to create the BFMatcher object using `cv2.BFMatcher()`. It takes two optional params. First one is `normType`. It specifies the distance measurement to be used. By default, it is `cv2.NORM_L2`. It is good for SIFT, SURF etc (`cv2.NORM_L1` is also there). For binary string based descriptors like ORB, BRIEF, BRISK etc, `cv2.NORM_HAMMING` should be used, which used Hamming distance as measurement. If ORB is using `VTA_K == 3 or 4`, `cv2.NORM_HAMMING2` should be used.
<div align="center">

  <img alt="BF" src="assets/bruteforce.jpg" />
 </div>
 
> Source: [Brute-Force Feature Matching](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html)






## 📝 CODE EXPLANATION


