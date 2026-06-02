"""image-movement: detect reuse of the same core image across user submissions.

Two-stage cascade: a permissive perceptual-hash filter (recall) followed by
geometric verification (precision). See detector.CascadeDetector.
"""

__version__ = "0.1.0"
