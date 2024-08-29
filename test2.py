import os
import cv2
import torch
import numpy as np
import gc
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor


class VideoSegmentation:
    def __init__(self, checkpoint_path, model_cfg_path, frames, mask_output_path, video_output_path, fps=30):
        self.checkpoint_path = checkpoint_path
        self.model_cfg_path = model_cfg_path
        self.frames = frames  # Now frames are passed directly as a list
        self.mask_output_path = mask_output_path
        self.predictor = self.get_predictor()
        self.inference_state = None

        # video:
        self.video_output_path = video_output_path
        self.fps = fps

        # Use bfloat16 and TF32 settings if applicable
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def get_predictor(self):
        """Load and return the SAM2 video predictor."""
        return build_sam2_video_predictor(self.model_cfg_path, self.checkpoint_path)

    def initialize_image_embedder_on_video_frames(self):
        """Initialize the image embedder on the video frames."""
        self.inference_state = self.predictor.init_state(video_path=self.frames)  # Passing frames directly
        return self.inference_state

    def get_mask_from_input_points(self, pos_input_points, neg_input_points=None, input_bounding_boxes=None, dynamic_ann_frame_id=-1, dyn_box=None):
        """Get the mask from the input points."""
        if neg_input_points is None:
            if pos_input_points is not None:
                points = np.array(pos_input_points, dtype=np.float32)
                labels = np.array([1 for _ in range(len(pos_input_points))], np.int32)
        else:
            raise NotImplementedError("Negative input points handling not implemented")

        ann_frame_idx = 0
        if pos_input_points is not None:
            ann_obj_id = len(points)
            with torch.amp.autocast('cuda'):
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )
        else:
            for i in range(len(input_bounding_boxes)):
                ann_obj_id = i
                with torch.amp.autocast('cuda'):
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        box=input_bounding_boxes[i],
                    )

        if dynamic_ann_frame_id >= 0:
            with torch.amp.autocast('cuda'):
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=dynamic_ann_frame_id,
                        obj_id=len(input_bounding_boxes) + 1 if len(input_bounding_boxes) is not None else 1,
                        box=dyn_box,
                    )

        plt.figure(figsize=(12, 8))
        plt.title(f"Frame {ann_frame_idx}")
        plt.imshow(self.frames[ann_frame_idx])  # Display the frame from the list of frames
        self.show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        plt.savefig("first_frame_masks.png")

    def track_segments(self, batch_size=8):
        """Track, crop, and save segments throughout the video."""
        video_segments = {}
        num_frames = len(self.frames)
        num_batches = (num_frames + batch_size - 1) // batch_size

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            torch.cuda.empty_cache()

        vis_frame_stride = 1
        plt.close("all")

        first_frame = self.frames[0]
        h, w = first_frame.shape[0], first_frame.shape[1]

        os.makedirs(os.path.join(root, "output_cropped"), exist_ok=True)
        os.makedirs(os.path.join(root, "output_mask"), exist_ok=True)
        os.makedirs(os.path.join(root, "binary_mask"), exist_ok=True)

        for batch_idx in range(num_batches):
            start_frame_idx = batch_idx * batch_size
            end_frame_idx = min((batch_idx + 1) * batch_size, num_frames)
            with torch.no_grad():
                for out_frame_idx in range(start_frame_idx, end_frame_idx, vis_frame_stride):
                    original_frame = self.frames[out_frame_idx]
                    plt.figure(figsize=(6, 4))
                    plt.title(f"Frame {out_frame_idx}")
                    plt.imshow(original_frame)

                    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                        self.show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                        plt.savefig(os.path.join(root, "output_mask", f"{out_frame_idx}.png"))

                        binary_mask = (out_mask.astype(np.uint8) * 255)
                        if binary_mask.ndim > 2:
                            binary_mask = binary_mask.squeeze()
                        if binary_mask.max() == 0:
                            print(f"No object found in frame {out_frame_idx}")
                            continue

                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        if contours:
                            sort_contour_area = []
                            for c in contours:
                                x, y, w, h = cv2.boundingRect(c)
                                if (cv2.contourArea(c)):
                                    sort_contour_area.append((c, cv2.contourArea(c)))
                            sort_contour_area.sort(key=lambda x: x[1], reverse=True)

                            x, y, w, h = cv2.boundingRect(sort_contour_area[0][0])

                            cropped_frame = original_frame[y:y + h, x:x + w]
                            cropped_frame_path = os.path.join(root, "output_cropped", f"cropped_{out_frame_idx}.png")

                            cropped_im = Image.fromarray(cropped_frame)
                            cropped_im.save(cropped_frame_path)
                    plt.close()

                    torch.cuda.empty_cache()
                    gc.collect()

def run_segmentation(frames, input_array):
    checkpoint_path = os.path.join(os.getcwd(), "segment-anything-2", "checkpoints", "sam2_hiera_base_plus.pt")
    model_cfg_path = "sam2_hiera_b+.yaml"
    mask_output_path = os.path.join(os.getcwd(), "segment-anything-2", "output_mask")
    video_output_path = os.path.join(os.getcwd(), "segment-anything-2", "output_video")

    segmentation = VideoSegmentation(checkpoint_path, model_cfg_path, frames, mask_output_path, video_output_path)

    segmentation.initialize_image_embedder_on_video_frames()
    segmentation.get_mask_from_input_points(pos_input_points=None, input_bounding_boxes=input_array, dyn_box=None)
    segmentation.track_segments()


def parse_input_array(file_path):
    try:
        with open(file_path, "r") as f:
            line = f.readline().strip()
            coords = list(map(int, line.split()))
            if len(coords) == 4:
                return [np.array(coords, dtype=np.int16)]
            else:
                print(f"Warning: The line in {file_path} does not have exactly four coordinates: {line}")
                return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def load_video_frames(video_path):
    """Load video frames as a list of NumPy arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

if __name__ == "__main__":
    video_path = '/home/lstajer/sam2_exp/segment-anything-2/video/f380c40d-ac20-4a45-ba1a-722554572985_1_1_1_2_2023-08-30T07-08-18.mp4'
    base_directory = os.path.join(os.getcwd(), "segment-anything-2", "video_frames_points")

    # Load video frames
    frames = load_video_frames(video_path)

    for root, dirs, files in os.walk(base_directory):
        txt_files = [file for file in files if file.endswith(".txt")]
        if txt_files:
            txt_file = txt_files[0]
            txt_file_path = os.path.join(root, txt_file)

            input_array = parse_input_array(txt_file_path)
            if input_array is not None:
                print(f"Processing {root} with input_array {input_array}")
                run_segmentation(frames, input_array)