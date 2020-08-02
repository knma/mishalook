from __future__ import print_function

import os, sys, tempfile
from types import SimpleNamespace

import math
import cv2, glob
import numpy as np
import torch, torchvision
from PIL import Image
from IPython.display import clear_output

torch_is_valid = torch.__version__.startswith("1.5")
if torch_is_valid:
  import numpy as np
  import cv2
  import torch
  import detectron2

  from detectron2.utils.logger import setup_logger
  setup_logger()
  from google.colab.patches import cv2_imshow
  from detectron2 import model_zoo
  from detectron2.engine import DefaultPredictor
  from detectron2.config import get_cfg
  from detectron2.utils.visualizer import Visualizer, ColorMode
  from detectron2.data import MetadataCatalog
  coco_metadata = MetadataCatalog.get("coco_2017_val")

  pointrend_dir = "detectron2_repo/projects/PointRend/"
  if pointrend_dir not in sys.path:
    sys.path.insert(0, pointrend_dir)
  import point_rend

  from ipywidgets import interact, interactive, fixed, interact_manual, Layout
  import ipywidgets as widgets
  from IPython.display import display

detectron_kps = {
  "nose": 0,
  "left_eye": 1,
  "right_eye": 2,
  "left_ear": 3,
  "right_ear": 4,
  "left_shoulder": 5,
  "right_shoulder": 6,
  "left_elbow": 7,
  "right_elbow": 8,
  "left_wrist": 9,
  "right_wrist": 10,
  "left_hip": 11,
  "right_hip": 12,
  "left_knee": 13,
  "right_knee": 14,
  "left_ankle": 15,
  "right_ankle": 16
}

detectron_kp_names = tuple(detectron_kps.keys())
detectron_kps = SimpleNamespace(**detectron_kps)



class MishaLook():
  def __init__(self, verbosity=0):
    self.verbosity = verbosity

    self.commands = SimpleNamespace(
      clone_detectron2 = "git clone --branch v0.2 https://github.com/facebookresearch/detectron2 detectron2_repo",
      pycocotools = "pip install pyyaml==5.1 pycocotools>=2.0.1",
      torch = "pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html",
      detectron2 = "pip install detectron2==0.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html",
    )

    self.cfg = SimpleNamespace(
      image_dir = "/content/drive/My Drive/LookPhotos/original",
      masks_dir = "/content/drive/My Drive/LookPhotos/masks",
      aligned_dir = "/content/drive/My Drive/LookPhotos/aligned",
      kp_predictor_image_height = 512,
      aligned_image_size = 1024,
      height_ratio = 0.9,
      vertical_shift = 0,
      horizontal_shift = 0,
      replace_background = True,
      change_aspect = False,
      target_aspect=0.75,
      background_color = '#ffffff',
      preview_image_index=0,
      dilate=0,
      erode=0,
      blur=0,
      n_images_limit=1000,
      jpeg_quality=99,
      preview_grid_range=[0,0],
      make_grid = True,
      grid_width_max=1500,
      grid_width_item=180,
    )

    if not torch_is_valid:
      print("Preparing.. It may take a few minutes.")
      self.setup_torch()
      self.setup_detectron()
      if self.verbosity == 0:
        clear_output()
      print("Please restart runtime")
    else:
      cfg = get_cfg()
      point_rend.add_pointrend_config(cfg)
      cfg.merge_from_file("detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
      cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
      cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl"
      self.mask_predictor  = DefaultPredictor(cfg)

      dcfg = get_cfg()
      dcfg.merge_from_file("detectron2_repo/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
      dcfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
      dcfg.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
      self.kp_predictor = DefaultPredictor(dcfg)

      clear_output()
      print("Ready to continue")

    self.configured = False

  def get_keypoints(self, images):
    keypoints = {}
    for im_name, image in images.items():
      outputs = self.kp_predictor(image)
      keypoints[im_name] = outputs['instances'].to('cpu')
    return keypoints

  def get_masks(self, images):
    masks = {}
    for im_name, image in images.items():
      outputs = self.mask_predictor(image)
      masks[im_name] = outputs['instances'].to('cpu')
    return masks

  def alpha_blend(self, img1, img2, mask):
    if mask.ndim==3 and mask.shape[-1] == 3:
      alpha = mask / 255
    else:
      alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended

  def align_images(self, images, keypoints, boxes, aligned_image_size, align_pad=0, height_ratio=1, vertical_shift=0, horizontal_shift=0, target_aspect=None):
    def rotate(point, origin, angle):
      ox, oy = origin
      px, py = point[:, 0], point[:, 1]
      qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
      qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
      return np.stack([qx, qy], axis=1)

    dkps = detectron_kps
    aligned_size = aligned_image_size
    aligned_images = {}
    aligned_keypoints = {}
    aligned_boxes = {}
    for im_name, image in images.items():
      kpts = keypoints[im_name][:,:2].copy()

      box = boxes[im_name].copy()
      box_height = box[3] - box[1]

      aspect = image.shape[1] / image.shape[0]
      if target_aspect is not None:
        aspect = target_aspect

      height = box_height * height_ratio

      mid = (box[:2] + box[2:4]) * 0.5

      shift = np.array([horizontal_shift * image.shape[1], vertical_shift * image.shape[0]])
      mid += shift

      left_top = mid - height //2
      left_top[0] = mid[0] - (height*aspect) //2

      scale = aligned_size / height
      scaled_size = [image.shape[1] * scale, image.shape[0] * scale]
      scaled_size = tuple([int(v) for v in scaled_size])

      tm = np.float32([ [scale,0,0], [0,scale,0] ])
      bg_color = self.hexToRGB(self.cfg.background_color)
      bg_color[0], bg_color[2] = bg_color[2], bg_color[0]
      bg_color = tuple(bg_color)
      scaled_image = cv2.warpAffine(image, tm, scaled_size, flags=cv2.INTER_CUBIC, borderValue=bg_color)

      left_top *= scale
      tm = np.float32([ [1,0,-left_top[0]], [0,1,-left_top[1]] ])

      out_size = (int(aligned_size * aspect), aligned_size)
      aligned_images[im_name] = cv2.warpAffine(scaled_image, tm, out_size, flags=cv2.INTER_CUBIC, borderValue=bg_color)

      kpts *= scale
      kpts -= left_top
      kpts = kpts[...,:2]

      box *= scale
      box[:2] = box[:2] - left_top
      box[2:4] = box[2:4] - left_top

      aligned_boxes[im_name] = box
      aligned_keypoints[im_name] = kpts[...,:2]

    return aligned_images, aligned_keypoints, aligned_boxes

  def configure(self):
    self.get_image_paths()

    if len(self.image_paths) < 1:
      return

    self.w_height_ratio = widgets.BoundedFloatText(
        value=self.cfg.height_ratio,
        description='Height ratio:',
        disabled=False,
        min = 0.5,
        max = 1.5,
        step=0.01,
        style = {'description_width': 'initial'}
    )
    display(self.w_height_ratio)

    self.w_vertical_shift = widgets.BoundedFloatText(
        value=self.cfg.vertical_shift,
        description='Vertical shift:',
        disabled=False,
        min = -0.3,
        max = 0.3,
        step=0.001,
        style = {'description_width': 'initial'}
    )
    display(self.w_vertical_shift)

    self.w_horizontal_shift = widgets.BoundedFloatText(
        value=self.cfg.horizontal_shift,
        description='Horizontal shift:',
        disabled=False,
        min = -0.3,
        max = 0.3,
        step=0.001,
        style = {'description_width': 'initial'}
    )
    display(self.w_horizontal_shift)

    self.w_replace_background = widgets.Checkbox(
        value=self.cfg.replace_background,
        description='Replace background',
        disabled=False,
        indent=False,
        style = {'description_width': 'initial'}
    )
    display(self.w_replace_background)

    self.w_colorpicker = widgets.ColorPicker(
        concise=False,
        description='Background color:',
        value=self.cfg.background_color,
        disabled=False,
        style = {'description_width': 'initial'}
    )
    display(self.w_colorpicker)

    self.w_change_aspect = widgets.Checkbox(
        value=self.cfg.change_aspect,
        description='Change aspect ratio',
        disabled=False,
        indent=False,
        style = {'description_width': 'initial'}
    )
    display(self.w_change_aspect)

    self.w_target_aspect = widgets.BoundedFloatText(
        value=self.cfg.target_aspect,
        description='Aspect ratio:',
        disabled=False,
        min = 0.5,
        max = 1.5,
        step=0.01,
        style = {'description_width': 'initial'}
    )
    display(self.w_target_aspect)

    self.w_dilation = widgets.BoundedIntText(
        value=self.cfg.dilate,
        min = 0,
        max = 20,
        step=1,
        description=f'Mask dilation:',
        disabled=False,
        style = {'description_width': 'initial'}
    )
    display(self.w_dilation)

    self.w_erosion = widgets.BoundedIntText(
        value=self.cfg.erode,
        min = 0,
        max = 20,
        step=1,
        description=f'Mask erosion:',
        disabled=False,
        style = {'description_width': 'initial'}
    )
    display(self.w_erosion)

    self.w_blur = widgets.BoundedIntText(
        value=self.cfg.blur,
        min = 0,
        max = 20,
        step=1,
        description=f'Mask blur:',
        disabled=False,
        style = {'description_width': 'initial'}
    )
    display(self.w_blur)

    self.w_preview_image_index = widgets.BoundedIntText(
        value=self.cfg.preview_image_index,
        min = 0,
        max = len(self.image_paths)-1,
        step=1,
        description=f'Preview image index (0..{len(self.image_paths)-1}):',
        disabled=False,
        style = {'description_width': 'initial'}
    )
    display(self.w_preview_image_index)

    self.w_make_grid = widgets.Checkbox(
        value=self.cfg.make_grid,
        description='Make preview grid',
        disabled=False,
        indent=False,
        style = {'description_width': 'initial'}
    )
    display(self.w_make_grid)

    self.w_preview_grid_range = widgets.IntRangeSlider(
        value=self.cfg.preview_grid_range,
        min=0,
        max=len(self.image_paths),
        step=1,
        description='Grid range:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        style = {'description_width': 'initial'}
    )
    display(self.w_preview_grid_range)

    self.btn_process_single = widgets.Button(
        description='Preview',
        disabled=False,
        button_style='',
        icon='check'
    )
    self.btn_process_single.on_click(self.process_preview)
    display(self.btn_process_single)

  def configure_all(self):
    self.get_image_paths()

    if not self.configured:
      print("Please configure a single image first")
      return

    self.w_image_size = widgets.BoundedIntText(
        value=self.cfg.aligned_image_size,
        min = 512,
        max = 6144,
        step=1,
        description='Result image size:',
        disabled=False,
        style = {'description_width': 'initial'}
    )
    display(self.w_image_size)

    self.w_image_quality = widgets.BoundedIntText(
        value=self.cfg.jpeg_quality,
        min = 50,
        max = 100,
        step=1,
        description='Result image quality:',
        disabled=False,
        style = {'description_width': 'initial'}
    )
    display(self.w_image_quality)

    self.w_n_images_limit = widgets.BoundedIntText(
        value=len(self.image_paths),
        min = 1,
        max = 1000,
        step=1,
        description='Number of images to process::',
        disabled=False,
        style = {'description_width': 'initial'}
    )
    display(self.w_n_images_limit)

    self.btn_process_all = widgets.Button(
        description='Process All Images',
        disabled=False,
        button_style='',
        icon='check'
    )
    self.btn_process_all.on_click(self.process_all)
    display(self.btn_process_all)

  def hexToRGB(self, hexa):
   return[int(hexa[1:][i:i+2], 16) for i in (0, 2, 4)]

  def update_cfg(self):
    self.cfg.height_ratio = self.w_height_ratio.value
    self.cfg.vertical_shift = self.w_vertical_shift.value
    self.cfg.horizontal_shift = self.w_horizontal_shift.value
    self.cfg.height_ratio = self.w_height_ratio.value
    self.cfg.replace_background = self.w_replace_background.value
    self.cfg.make_grid = self.w_make_grid.value
    self.cfg.background_color = self.w_colorpicker.value
    self.cfg.preview_image_index = self.w_preview_image_index.value
    self.cfg.preview_grid_range = self.w_preview_grid_range.value
    self.cfg.erode = self.w_erosion.value
    self.cfg.dilate = self.w_dilation.value
    self.cfg.blur = self.w_blur.value
    self.cfg.change_aspect = self.w_change_aspect.value
    self.cfg.target_aspect = self.w_target_aspect.value

  def process_preview(self, *args):
    self.update_cfg()
    image_path = self.image_paths[self.cfg.preview_image_index]
    self.configured = True

    grid_img = None
    preview_single_img = None

    print("\nProcessing...")

    preview_single_img = self.process_single(image_path)
    print(f"Preview image: {os.path.basename(image_path)}")

    if self.cfg.preview_grid_range[1] > 0 and self.cfg.make_grid:
      grid = []
      pos = {}
      cell_size = None
      for i in range(len(self.image_paths)):
        if i < self.cfg.preview_grid_range[0] or i >= self.cfg.preview_grid_range[1]:
          continue
        img = self.process_single(self.image_paths[i], is_preview=True, aligned_only=True)
        if img if not None:
          grid.append(img)
          if cell_size is None:
            aspect = img.shape[1] / img.shape[0]
            cell_size = (self.cfg.grid_width_item, int(self.cfg.grid_width_item/aspect))
        print(f"Grid {i+1}: {os.path.basename(self.image_paths[i])}")
      for i, img in enumerate(grid):
        grid[i] = cv2.resize(img, cell_size)
      n_cols_max = self.cfg.grid_width_max // self.cfg.grid_width_item
      x_max = 0
      y_max = 0
      for i, img in enumerate(grid):
        row = i // n_cols_max
        col = i % n_cols_max
        x = col * img.shape[1]
        y = row * img.shape[0]
        pos[i] = (x,y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)
      v_lines = set()
      h_lines = set()
      if x_max>0:
        grid_w = x_max+cell_size[0]
        grid_h = y_max+cell_size[1]
        grid_img = np.ones((grid_h, grid_w, grid[0].shape[2]), grid[0].dtype) * 255
        for i, img in enumerate(grid):
          p = pos[i]
          v_lines.add(p[0])
          h_lines.add(p[1])
          grid_img[p[1]:p[1]+cell_size[1], p[0]:p[0]+cell_size[0]] = img
        v_lines.add(grid_w-1)
        h_lines.add(grid_h-1)
        for x in v_lines: 
          cv2.line(grid_img, (x,0), (x, grid_h-1), (150, 150, 150), thickness=1)
        for y in h_lines: 
          cv2.line(grid_img, (0,y), (grid_w-1, y), (150, 150, 150), thickness=1)

    clear_output()
    self.configure()

    if preview_single_img is not None:
        display(Image.fromarray(preview_single_img[...,::-1]))
        print()
    if grid_img is not None:
        display(Image.fromarray(grid_img[...,::-1]))


  def process_all(self, *args):
    self.cfg.aligned_image_size = self.w_image_size.value
    self.cfg.jpeg_quality = self.w_image_quality.value
    self.cfg.n_images_limit = self.w_n_images_limit.value
    clear_output()
    self.configure_all()

    os.makedirs(self.cfg.masks_dir, exist_ok=True)
    os.makedirs(self.cfg.aligned_dir, exist_ok=True)

    for i, image_path in enumerate(self.image_paths):
      if i >= self.cfg.n_images_limit:
        break
      print(f"{i+1} / {len(self.image_paths)}: {os.path.basename(image_path)}")
      self.process_single(image_path, is_preview=False)

    print("Done")

  def process_single(self, image_path, is_preview=True, aligned_only=False):
    interm = SimpleNamespace(**{
      'images': {},
      'base_names': {},
      'resized_images': {},
      'keypoints': {},
      'boxes': {},
      'aligned_images': {},
      'aligned_keypoints': {},
      'aligned_boxes': {},
      'masks': {},
      'out_scales': {},
    })

    for i, image_path in enumerate([image_path]):
      image = cv2.imread(image_path)
      im_name = f"img_{str(i).zfill(4)}" 
      interm.images[im_name] = image
      interm.base_names[im_name] = os.path.basename(image_path)

    # Resize images
    for im_name, image in interm.images.items():
      new_size = (
        int(image.shape[1] / image.shape[0] * self.cfg.kp_predictor_image_height),
        self.cfg.kp_predictor_image_height
      )
      interm.resized_images[im_name] = cv2.resize(image, new_size)

    # Predict and verify keypoints
    for im_name, im_keypoints in self.get_keypoints(interm.resized_images).items():
      all_keypoints = im_keypoints.get_fields()['pred_keypoints']
      all_boxes = im_keypoints.get_fields()['pred_boxes']
      if len(all_keypoints) < 1:
        print(f"Cannot process {interm.base_names[im_name]}")
        return None
      interm.keypoints[im_name] = all_keypoints[0].numpy()
      interm.boxes[im_name] = all_boxes.tensor[0].numpy()

    interm.aligned_images, interm.aligned_keypoints, interm.aligned_boxes = self.align_images(
      interm.resized_images,
      interm.keypoints,
      interm.boxes,
      self.cfg.kp_predictor_image_height,
      height_ratio = 1/self.cfg.height_ratio,
      target_aspect = self.cfg.target_aspect,
      vertical_shift=self.cfg.vertical_shift,
      horizontal_shift=self.cfg.horizontal_shift
    )

    for im_name, im_masks in self.get_masks(interm.aligned_images).items():
      classes = im_masks.get_fields()['pred_classes'].numpy()
      human_indices = np.where(classes == 0)[0]
      # if len(human_indices) == 1:
      if len(human_indices) < 1:
        print(f"Cannot process {interm.base_names[im_name]}")
        return None
      mask = im_masks.get_fields()['pred_masks'].numpy()[human_indices[0]]
      mask = mask[..., None] * 1.0
      mask = np.tile(mask.astype(np.uint8) * 255, (1,1,3))
      if self.cfg.erode > 0:
        mask = cv2.erode(
          mask,
          np.ones((3, 3), np.uint8),
          iterations=self.cfg.erode
        )
      if self.cfg.dilate > 0:
        mask = cv2.dilate(
          mask,
          np.ones((3, 3), np.uint8),
          iterations=self.cfg.dilate
        )

      mask = mask[..., 0] > 128
      interm.masks[im_name] = mask

    for im_name, image in interm.aligned_images.items():
      preview = interm.aligned_images[im_name].copy()

      for j, kp in enumerate(interm.aligned_keypoints[im_name]):
        kp_pos = tuple([int(v) for v in kp[:2]])
        cv2.circle(preview, kp_pos, 2, (0,0,255), -1)

      mask = interm.masks[im_name] * 255
      mask = np.tile(mask[...,None], (1,1,3)).astype(np.uint8)

      if self.cfg.blur > 0:
        # mask_sise = (mask.shape[1], mask.shape[0])
        # mask_half_sise = (mask.shape[1]//2, mask.shape[0]//2)
        # mask = cv2.resize(mask, mask_half_sise)
        for bi in range(self.cfg.blur):
          mask = cv2.blur(
            mask,
            (7,7)
          )
        # mask = cv2.resize(mask, mask_half_sise)

      interm.masks[im_name] = mask

      if is_preview:
        green = np.zeros_like(preview)
        green[...,1] = 255
        preview = self.alpha_blend(green, preview, mask)
        preview = np.clip(preview, 0, 255).astype(np.uint8)

        box = interm.aligned_boxes[im_name].reshape((2,2))
        cv2.rectangle(preview, (box[0,0], box[0,1]), (box[1,0], box[1,1]), (255, 0, 0), 1)

        img_aligned = image
        if self.cfg.replace_background:
          bg_color = self.hexToRGB(self.cfg.background_color)
          bg_color[0], bg_color[2] = bg_color[2], bg_color[0]
          bg = np.ones_like(preview) * np.array(bg_color)
          img_aligned = self.alpha_blend(bg, image, mask)

        if not aligned_only:
          preview = np.hstack([interm.resized_images[im_name], preview, img_aligned])
          return preview
          # pil_image = Image.fromarray(preview[...,::-1])
          # display(pil_image)
        else:
          return img_aligned

    if not is_preview:
      out_scale = 1
      for im_name, image in interm.images.items():
        out_scale = image.shape[0] / interm.aligned_images[im_name].shape[0]
        interm.keypoints[im_name] *= out_scale
        interm.boxes[im_name] *= out_scale

      interm.out_images, interm.out_keypoints, interm.out_boxes = self.align_images(
        interm.images,
        interm.keypoints,
        interm.boxes,
        self.cfg.aligned_image_size,
        height_ratio = 1/self.cfg.height_ratio,
        target_aspect = self.cfg.target_aspect,
        vertical_shift=self.cfg.vertical_shift,
        horizontal_shift=self.cfg.horizontal_shift
      )

      for im_name, out_image in interm.out_images.items():
        mask = interm.masks[im_name]
        mask = cv2.resize(mask, (out_image.shape[1], out_image.shape[0]))

        base_name = interm.base_names[im_name]
        base_name_noext = os.path.splitext(base_name)[0]

        out_mask_path = os.path.join(self.cfg.masks_dir, base_name_noext + ".png")
        cv2.imwrite(out_mask_path, mask)

        if self.cfg.replace_background:
          bg_color = self.hexToRGB(self.cfg.background_color)
          bg_color[0], bg_color[2] = bg_color[2], bg_color[0]
          bg = np.ones_like(out_image) * np.array(bg_color)
          out_image = self.alpha_blend(bg, out_image, mask)

        out_image_path = os.path.join(self.cfg.aligned_dir, base_name_noext + ".jpg")
        cv2.imwrite(out_image_path, out_image, [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality])

        # display(Image.fromarray(mask[...,::-1]))
        # display(Image.fromarray(out_image[...,::-1]))


  def get_image_paths(self):
    image_dir = self.cfg.image_dir
    self.image_paths = []
    for img_type in ('*.png', '*.jpg'):
        self.image_paths.extend(list(sorted(glob.glob(os.path.join(image_dir, img_type)))))

    print(f"Found {len(self.image_paths)} images")

   
  def runcmd(self, cmd):
    ftmp = tempfile.NamedTemporaryFile(suffix='.out', prefix='tmp', delete=False)
    fpath = ftmp.name
    if os.name=="nt":
      fpath = fpath.replace("/","\\")
    ftmp.close()
    os.system(cmd + " > " + fpath)
    data = ""
    with open(fpath, 'r') as file:
      data = file.read()
      file.close()
    os.remove(fpath)
    if self.verbosity > 1:
      print(data)
    return data

  def setup_torch(self):
    out = self.runcmd(self.commands.pycocotools)
    if self.verbosity > 0:
      print("pycocotools ok")    

    out = self.runcmd(self.commands.torch)
    if self.verbosity > 0:
      print("torch ok")

  def setup_detectron(self):
    if not os.path.isdir("detectron2_repo"):
      out = self.runcmd(self.commands.clone_detectron2)
      sys.path.insert(0, "detectron2_repo/projects/PointRend/")
      if self.verbosity > 0:
        print("detectron2_repo ok")

    out = self.runcmd(self.commands.detectron2)
    if self.verbosity > 0:
      print("detectron2 ok")    
