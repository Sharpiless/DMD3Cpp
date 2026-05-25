import torch
import torch.nn.functional as F


class InputProcessor:
    PATCH_SIZE = 14

    def __call__(
        self,
        image: torch.Tensor,                  # [B, 3, H, W], already normalized
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,   # [B, 3, 3]
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
    ):
        """
        Args:
            image:      [B, 3, H, W], torch.Tensor, already normalized
            extrinsics: optional, returned as-is
            intrinsics: [B, 3, 3], optional

        Returns:
            image_out:      [B, 3, H', W']
            extrinsics_out: same as input
            intrinsics_out: [B, 3, 3] after resize/crop correction
        """
        assert isinstance(image, torch.Tensor), f"image must be torch.Tensor, got {type(image)}"
        assert image.ndim == 4 and image.shape[1] == 3, \
            f"image must be [B,3,H,W], got {tuple(image.shape)}"

        B, C, orig_h, orig_w = image.shape

        if intrinsics is not None:
            assert isinstance(intrinsics, torch.Tensor), f"intrinsics must be torch.Tensor, got {type(intrinsics)}"
            assert intrinsics.shape == (B, 3, 3), \
                f"intrinsics must be [B,3,3], got {tuple(intrinsics.shape)}"

        K = intrinsics.clone() if intrinsics is not None else None

        # 1) boundary resize
        if process_res_method in ("upper_bound_resize", "upper_bound_crop"):
            scale = process_res / float(max(orig_h, orig_w))
        elif process_res_method in ("lower_bound_resize", "lower_bound_crop"):
            scale = process_res / float(min(orig_h, orig_w))
        else:
            raise ValueError(f"Unsupported process_res_method: {process_res_method}")

        resized_h = max(1, int(round(orig_h * scale)))
        resized_w = max(1, int(round(orig_w * scale)))

        if resized_h != orig_h or resized_w != orig_w:
            image = F.interpolate(
                image,
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=False,
            )
            K = self._resize_ixt(K, orig_w, orig_h, resized_w, resized_h)

        # 2) enforce divisibility by PATCH_SIZE
        if process_res_method.endswith("resize"):
            final_h = self._nearest_multiple(resized_h, self.PATCH_SIZE)
            final_w = self._nearest_multiple(resized_w, self.PATCH_SIZE)

            if final_h != resized_h or final_w != resized_w:
                image = F.interpolate(
                    image,
                    size=(final_h, final_w),
                    mode="bilinear",
                    align_corners=False,
                )
                K = self._resize_ixt(K, resized_w, resized_h, final_w, final_h)

        elif process_res_method.endswith("crop"):
            final_h = (resized_h // self.PATCH_SIZE) * self.PATCH_SIZE
            final_w = (resized_w // self.PATCH_SIZE) * self.PATCH_SIZE

            crop_top = max(0, (resized_h - final_h) // 2)
            crop_left = max(0, (resized_w - final_w) // 2)

            if final_h != resized_h or final_w != resized_w:
                image = image[:, :, crop_top:crop_top + final_h, crop_left:crop_left + final_w]
                K = self._crop_ixt(K, crop_left, crop_top)

        return image.contiguous(), extrinsics, K

    @staticmethod
    def _nearest_multiple(x: int, p: int) -> int:
        down = (x // p) * p
        up = down + p
        if down <= 0:
            return up
        return up if abs(up - x) <= abs(x - down) else down

    @staticmethod
    def _resize_ixt(K: torch.Tensor | None, orig_w: int, orig_h: int, new_w: int, new_h: int):
        if K is None:
            return None
        K = K.clone()
        sx = new_w / float(orig_w)
        sy = new_h / float(orig_h)

        # 与原实现一致：第0行按宽缩放，第1行按高缩放
        K[:, 0, :] *= sx
        K[:, 1, :] *= sy
        return K

    @staticmethod
    def _crop_ixt(K: torch.Tensor | None, crop_left: int, crop_top: int):
        if K is None:
            return None
        K = K.clone()
        K[:, 0, 2] -= crop_left
        K[:, 1, 2] -= crop_top
        return K