# Repository guidance

- Do not track source images, the contents of `raw-data/`, or generated images
  whose names contain the `original` or `overlay` tag. Masks and masked images may
  be tracked when they contain only the intended privacy-safe output.

- When implementing new logic for the next steps, always take into account that we don't have to repro every image from scratch if an image is processed for a certain step, we can skip that step for that specific image so if the new logic you are implementing is taking over from an already processed step. It should check the next step folder and check if an image processed or not if it was processed, it can skip that image with a note.

- For resource, heavy calculations, and logs and timer so that we can know how long each item took to process.

- This will be a water waviness regression model, which will process an incoming image and first preprocess the image to utilize the sea area and then check the sea area to calculate how much wave there is in the water.

- Location specific requirements: the waves are usually visible at the bottom part of the image so when we are positioning the processed image we should prefer rectangles closer to the bottom of the detected sea.

- This project is mainly for learning purposes, so keep the coat clean and keep the processes as separate as possible with information transparent.

- Each step up model training processes should work on the file that was the output of the last spec. Do not work on the original file for an intermediate step. Only the first step should work on the original file and then it's output should be used for the next step.

- Treat this as a pipeline so that there could be new files any time. When new files are processed the older files shouldn't be a repprocessed again if they already processed.

- Follow the repository's step naming convention for new pipeline files: use names such as `step_4_a_<purpose>.py`, `step_4_b_<purpose>.py`, and so on, keeping the stage order explicit.
