CIFAR-100 has 100 classes with 600 images each.

Each image is 32x32: 1024 pixels. Each pixel has an 3 values (RGB). We want to keep
the xy plane with 3 channels. RGB values are integers.

We probably want to convert to float and normalize? Normalize how? Subtract each
RGB value.
