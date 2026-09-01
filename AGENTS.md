# Repository guidance

- Do not track source images, the contents of `raw-data/`, or generated images
  whose names contain the `original` or `overlay` tag. Masks and masked images may
  be tracked when they contain only the intended privacy-safe output.

- When implementing new logic for the next steps, always take into account that we don't have to repro every image from scratch if an image is processed for a certain step, we can skip that step for that specific image so if the new logic you are implementing is taking over from an already processed step. It should check the next step folder and check if an image processed or not if it was processed, it can skip that image with a note.

- For resource, heavy calculations, and logs and timer so that we can know how long each item took to process.
