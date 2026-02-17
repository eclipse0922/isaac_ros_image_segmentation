# Timing metrics for SAM3 inference pipeline
# Published by sam3_node.py for benchmarking

std_msgs/Header header

# Per-stage timing in milliseconds
float64 cvbridge_ms           # ROS Image to OpenCV conversion
float64 preprocess_ms         # Image resize, pad, normalize to 1008x1008
float64 vision_encoder_ms     # Vision encoder inference
float64 text_encoder_ms       # Text encoder inference (0.0 if cache hit)
bool text_encoder_cache_hit   # Whether text features were cached
float64 decoder_ms            # Decoder inference (sum of all prompts)
int32 num_prompts             # Number of text prompts processed
float64 postprocess_ms        # Detection post-processing and publishing
float64 total_ms              # Total end-to-end latency

# Configuration
string backend                # "triton" or "pytorch"
string model_type             # "sam3" or "efficient_sam3"
