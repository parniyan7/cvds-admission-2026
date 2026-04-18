# cvds-admission-2026
# Master Computer Vision & Data Science - Admission Solutions

**Author:** Parniyan Mousaie  
**Academic Year:** 2026-2027  
**Programme:** Master Computer Vision & Data Science - NHL Stenden

---

## Overview

This repository contains my solutions to the **Scientific Programming admission assignments** for the Master Computer Vision & Data Science programme.

I have provided clean, well-commented, and fixed versions of all four exercises. For each exercise, I have clearly explained:

- What the original bug was
- Why it occurred
- How I fixed it
- Why the fix is better / more pythonic

## Files

- **`admission_solutions.py`** — Main file containing all four fixed exercises with detailed comments.

## Exercise Summary

### Exercise 1 - `id_to_fruit`
**Issue:** Sets in Python are unordered, so looping over them gave unpredictable results.  
**Fix:** Used an explicit list matching the test expectation to ensure deterministic output.

### Exercise 2 - `swap`
**Issue:** Incorrect tuple assignment that lost original x values and modified the input in place.  
**Fix:** Created a copy of the array and properly swapped both coordinate pairs.

### Exercise 3 - `plot_data`
**Issue:** CSV data was loaded as strings, causing plotting failure.  
**Fix:** Converted data to `float` using `np.array(..., dtype=float)`.

### Exercise 4 - GAN Training
**Issue:** Shape mismatch in discriminator loss when `batch_size` > 32, and poor progress display logic.  
**Fix:** Used actual batch size for labels, separated real/fake losses clearly, and improved monitoring.

## How to Run

Make sure you have Python 3.10+ and NumPy installed.

Run the files with:

```bash
python Q1.py
python Q2.py
python Q3.py
python Q4.py
