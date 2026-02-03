import gym
from gym.spaces import flatten_space, flatten


# Optional resizers
import numpy as np
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

def _resize_hwc(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """
    Resize HxWxC image to (H', W', C) without changing dtype/range.
    Supports cv2, PIL, or pure NumPy resizing.
    If img is float32, it will be scaled to [0,1] if not already in that range.
    If img is uint8, it will be resized as-is.

    Args:
        img (np.ndarray): Input image in HxWxC format.
        hw (tuple[int, int]): Target height and width (H', W').
    Returns:
        np.ndarray: Resized image in H'xW'xC format.
    """
    H, W = hw
    if _HAS_CV2:
        # cv2 wants (W, H)
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    
    if _HAS_PIL:
        pil = Image.fromarray(img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8))
        pil = pil.resize((W, H), resample=Image.Resampling.BILINEAR)
        out = np.asarray(pil)
        if img.dtype != np.uint8:
            # If original was float, map back to float [0,1]
            out = out.astype(np.float32) / 255.0
        return out
    
    # Pure NumPy (simple nearest neighbor)
    y_idx = (np.linspace(0, img.shape[0] - 1, H)).astype(np.int32)
    x_idx = (np.linspace(0, img.shape[1] - 1, W)).astype(np.int32)
    return img[y_idx][:, x_idx]    

class SERLObsWrapper(gym.ObservationWrapper):
    """
    Observation wrapper for SERL environments.
    Flattens the 'state' space and resizes images to a target height and width.
    Supports both uint8 and float32 images, with optional normalization.
    The observation space is a Dict with 'state' and resized image spaces.

    Args:
        env (gym.Env): The environment to wrap.
        target_hw (tuple[int, int]): Target height and width for resized images.
        img_dtype (np.dtype): Data type for images, either np.uint8 or np.float32.
        normalize (bool): If True, scales float32 images to [0,1].
        image_parent_key (str): Key in the observation dict where images are stored.    

        Defaults to "images".
    Returns:
        gym.spaces.Dict: The new observation space with flattened state and resized images.
    """

    def __init__(
        self,
        env,
        target_hw=(128, 128),        # (H, W) for resized images
        img_dtype=np.uint8,          # np.uint8 for [0..255], or np.float32 for [0..1]
        normalize=False,             # if True and img_dtype=float32, scale to [0,1]
        image_parent_key="images",   # where images live in the original obs dict
    ):
        super().__init__(env)
        assert isinstance(self.env.observation_space, gym.spaces.Dict), \
            "Expected Dict observation_space with keys {'state', 'images'}"
        
         # ---- Build new observation_space ----
        base_space = self.env.observation_space
        assert "state" in base_space.spaces, "Missing 'state' in observation_space"
        assert image_parent_key in base_space.spaces, f"Missing '{image_parent_key}' in observation_space"
        img_space_dict = base_space.spaces[image_parent_key]
        assert isinstance(img_space_dict, gym.spaces.Dict), \
            f"'{image_parent_key}' must be a Dict of image spaces"

        # Flattened state space
        state_space = flatten_space(base_space.spaces["state"])


        # Image spaces (resized)
        H, W = target_hw
        image_spaces = {}
        for k, sp in img_space_dict.spaces.items():
            # Assume HWC input; preserve channel count
            if hasattr(sp, "shape") and sp.shape is not None:
                if len(sp.shape) != 3:
                    raise ValueError(f"Image space '{k}' must be HxWxC; got shape {sp.shape}")
                C = sp.shape[-1]
            else:
                raise ValueError(f"Image space '{k}' missing shape")

            if img_dtype == np.uint8:
                low, high = 0, 255
            elif img_dtype == np.float32:
                low, high = 0.0, 1.0 if normalize else float(getattr(sp, "high", 1.0))
            else:
                raise ValueError("img_dtype must be np.uint8 or np.float32")

            image_spaces[k] = gym.spaces.Box(
                low=low,
                high=high,
                shape=(H, W, C),
                dtype=img_dtype,
            )

        # Final Dict space: {'state': ..., 'front': Box(...), 'wrist': Box(...), ...}
        self.observation_space = gym.spaces.Dict({
            "state": state_space,
            **image_spaces
        })            
        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "state": flatten_space(self.env.observation_space["state"]),
        #         **(self.env.observation_space["images"]),
        #     }
        # )

        # Store config
        self._target_hw = target_hw
        self._img_dtype = img_dtype
        self._normalize = normalize
        self._image_parent_key = image_parent_key

    # def observation(self, obs):
    #     obs = {
    #         "state": flatten(self.env.observation_space["state"], obs["state"]),
    #         **(obs["images"]),
    #     }
    #     return obs

    def observation(self, obs):
        # Flatten state using original (pre-flatten) state space definition
        flat_state = flatten(self.env.observation_space.spaces["state"], obs["state"])

        # Pull original images dict
        imgs = obs[self._image_parent_key]

        # Resize & cast each image to match observation_space spec
        out = {"state": flat_state}
        for k, sp in self.observation_space.spaces.items():
            if k == "state":
                continue
            img = imgs[k]
            # Ensure HWC
            if img.ndim != 3:
                raise ValueError(f"Image '{k}' must be HxWxC; got shape {img.shape}")

            # If float32 images in [0,1] but we want uint8, scale up before resize for best quality
            want_uint8 = (self._img_dtype == np.uint8)
            if want_uint8:
                if img.dtype != np.uint8:
                    # Assume 0..1 range; if 0..255 float, clip and cast
                    img = np.clip(img, 0.0, 1.0) if img.max() <= 1.0 else np.clip(img/255.0, 0.0, 1.0)
                    img = (img * 255.0 + 0.5).astype(np.uint8)
                resized = _resize_hwc(img, self._target_hw).astype(np.uint8)
            else:
                # float32 output
                if img.dtype == np.uint8:
                    if self._normalize:
                        img = img.astype(np.float32) / 255.0
                    else:
                        img = img.astype(np.float32)  # keep 0..255 range if you really want that
                else:
                    img = img.astype(np.float32)
                    if self._normalize and img.max() > 1.0:
                        img = img / 255.0
                resized = _resize_hwc(img, self._target_hw).astype(np.float32)
            
            out[k] = resized

        return out