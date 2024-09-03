import os
import torch
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sam2.build_sam import build_sam2_video_predictor
import gc


class VideoSegmentation:
    def __init__(self, frames_path, mask_output_path, video_output_path, model, fps = 30):
        self.frames_path = frames_path
        self.mask_output_path = mask_output_path
        self.predictor = model
        self.video_output_path = video_output_path
        self.fps = fps

        self.inference_state = None
        self.frame_names = None

        # Use bfloat16 and TF32 settings if applicable
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        """Display the mask with a specific color on the provided axis."""
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=200):
        """Display points with positive and negative labels on the provided axis."""
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def show_box(self, box, ax):
        """Draw a bounding box on the provided axis."""
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


    def get_points_from_boxes(self, boxes):
        """Convert bounding boxes to points."""
        return [[int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)] for box in boxes]

    def initialize_image_embedder_on_video_frames(self):
        """Initialize the image embedder on the video frames."""
        if isinstance(self.frames_path,str):
            self.frame_names = [
                p for p in os.listdir(self.frames_path)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".npy"]
            ]
            self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        else:
            assert all(isinstance(arr, np.ndarray) for arr in self.frames_path), "All elements in the list should be NumPy arrays."
            assert self.frames_path[0].ndim == 3, "The first element in the list is not a 3D array."
            self.frame_names = [f"{i + 1}.jpg" for i in range(len(self.frames_path))]
       
        self.inference_state = self.predictor.init_state(video_path=self.frames_path, offload_video_to_cpu=True, frame_names=self.frame_names)
        return self.inference_state, self.frame_names

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
                        obj_id=len(input_bounding_boxes)+1 if len(input_bounding_boxes) is not None else 1,
                        box=dyn_box,
                    )

        # plt.figure(figsize=(12, 8))
        # plt.title(f"Frame {ann_frame_idx}")
        
        # if not os.path.join(self.frames_path, self.frame_names[ann_frame_idx]).endswith('.npy'):
        #     plt.imshow(Image.open(os.path.join(self.frames_path, self.frame_names[ann_frame_idx])))
        # else:
        #     plt.imshow(np.load(os.path.join(self.frames_path, self.frame_names[ann_frame_idx])))
        # # self.show_points(points, labels, plt.gca())
        # self.show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        # plt.savefig("first_frame_masks.jpg")

    def track_segments(self, input_path, batch_size=8):
        """Track, crop, and save segments throughout the video."""
        video_segments = {}
        num_frames = len(self.frame_names)
        num_batches = (num_frames + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            del out_obj_ids, out_mask_logits
            
        self.predictor.reset_state(self.inference_state)
        del self.predictor    
        
        gc.collect()
        torch.cuda.empty_cache()

        vis_frame_stride = 1
        plt.close("all")

        if isinstance(self.frames_path,str):
            if not os.path.join(self.frames_path, self.frame_names[0]).endswith('.npy'):
                first_frame = Image.open(os.path.join(self.frames_path, self.frame_names[0]))
                h, w = first_frame.size
            else:
                first_frame = np.load(os.path.join(self.frames_path, self.frame_names[0]))
                w, h = first_frame.shape[0], first_frame.shape[1]
                os.makedirs(os.path.join(self.frames_path, "output_cropped"), exist_ok=True)
                os.makedirs(os.path.join(self.frames_path, "output_mask"), exist_ok=True)
        else:
            h, w = self.frames_path[0].shape[0], self.frames_path[0].shape[1]
            os.makedirs(os.path.join(input_path, "output_cropped"), exist_ok=True)
            os.makedirs(os.path.join(input_path, "output_mask"), exist_ok=True)

        for batch_idx in range(num_batches):
            start_frame_idx = batch_idx * batch_size
            end_frame_idx = min((batch_idx + 1) * batch_size, num_frames)

           
            with torch.no_grad():
                for out_frame_idx in range(start_frame_idx, end_frame_idx, vis_frame_stride):
                    # Load original frame1
                    if isinstance(self.frames_path,str):
                        if os.path.join(self.frames_path, self.frame_names[out_frame_idx]).endswith('.npy'):
                            original_frame = np.load(os.path.join(self.frames_path, self.frame_names[out_frame_idx]))
                        else:
                            original_frame = Image.open(os.path.join(self.frames_path, self.frame_names[out_frame_idx]))
                    else:
                        original_frame = self.frames_path[out_frame_idx]
                    
                    plt.figure(figsize=(6, 4))
                    plt.title(f"Frame {out_frame_idx}")
                    plt.imshow(original_frame)

                    # Get mask and determine bounding box
                    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                        self.show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                        plt.savefig(os.path.join(input_path,"output_mask",f"{out_frame_idx}.jpg"))

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
                                    sort_contour_area.append((c,cv2.contourArea(c)))
                            sort_contour_area.sort(key = lambda x:x[1] , reverse = True)

                            x, y, w, h = cv2.boundingRect(sort_contour_area[0][0])

                            # Crop the original frame using the bounding box
                            if os.path.join(input_path, self.frame_names[out_frame_idx]).endswith('.npy') or isinstance(original_frame, np.ndarray):
                                cropped_frame = original_frame[y:y + h, x:x + w]
                                cropped_frame = Image.fromarray(cropped_frame)
                            else:
                                cropped_frame = original_frame.crop((x, y, x + w, y + h))
                            
                            cropped_frame_path = os.path.join(input_path, "output_cropped", f"cropped_{out_frame_idx}.jpg")
                            cropped_frame.save(cropped_frame_path)
                    plt.close()

                    torch.cuda.empty_cache()
                    gc.collect()

def run_segmentation(file_path, input_array, model, input_path=None):
    frames_path = file_path
    mask_output_path =os.path.join(os.getcwd(),"segment-anything-2","output_mask")
    video_output_path =os.path.join(os.getcwd(),"segment-anything-2", "output_video")

    segmentation = VideoSegmentation(frames_path, mask_output_path, video_output_path, model)
    
    segmentation.initialize_image_embedder_on_video_frames()
    segmentation.get_mask_from_input_points(pos_input_points=None, input_bounding_boxes=input_array, dyn_box=None)
    segmentation.track_segments(input_path)

    print(f"{torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB allocated, {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB reserved")

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
    
def main():
    # base_directory = os.path.join(os.getcwd(),"segment-anything-2", "video_frames_np") # RENAME FOR PATH!!
    # base_directory = os.path.join(os.getcwd(),"segment-anything-2", "video_frames_points") # RENAME FOR PATH!!

    checkpoint_path = os.path.join(os.getcwd(),"segment-anything-2", "checkpoints", "sam2_hiera_large.pt")  
    model_cfg_path = "sam2_hiera_l.yaml"

    model = build_sam2_video_predictor(model_cfg_path, checkpoint_path)

    list_input_dir = os.path.join(os.getcwd(),"segment-anything-2", "vid_npz")
    for root, _, files in os.walk(list_input_dir):
        txt_files = [file for file in files if file.endswith(".txt")]
        if txt_files:
            txt_file = txt_files[0]  
            txt_file_path = os.path.join(root, txt_file)

            input_array = parse_input_array(txt_file_path)

            npz_file = np.load(os.path.join(root,files[0]))
            frames_array = npz_file['frames']

            if input_array is not None:
                print(f"Processing {root} with input_array {input_array}")
                run_segmentation(frames_array, input_array, model, input_path=list_input_dir)
    # img_list = os.path.join(os.getcwd(),"segment-anything-2", "image_list.npz")
    # run_segmentation(root, input_array, model)

if __name__ == "__main__":
    main()
