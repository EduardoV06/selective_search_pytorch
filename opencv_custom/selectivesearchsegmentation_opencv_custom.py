import os
import ctypes
import torch

import importlib.resources as resources
from pathlib import Path
import opencv_custom

so_path = resources.files(opencv_custom).joinpath("selectivesearchsegmentation_opencv_custom_.so")


class SelectiveSearchOpenCVCustom(torch.nn.Module):
    def __init__(self, preset = 'fast', remove_duplicate_boxes = False, lib_path = 'selectivesearchsegmentation_opencv_custom_.so', max_num_rects = 4096, max_num_planes = 16, max_num_bit = 64, base_k = 0, inc_k = 0, sigma = 0):
        super().__init__()
        # self.bind = ctypes.CDLL(lib_path)
        self.bind = ctypes.CDLL(str(so_path))
        self.bind.process.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ctypes.c_bool, ctypes.c_int64
        ]
        self.bind.process.restype = ctypes.c_int
        self.bind_num_rects = ctypes.c_int()
        self.bind_num_planes = ctypes.c_int()
        self.preset = preset
        self.remove_duplicate_boxes = remove_duplicate_boxes
        self.base_k = base_k
        self.inc_k = inc_k
        self.sigma = sigma
        self.max_num_rects = max_num_rects
        self.max_num_planes = max_num_planes
        self.max_num_bit = max_num_bit
        self.byte_nonzero = [[i for i in range(s.bit_length()) if s & (1 << i)] for s in range(256)]
        self.bit_nonzero = lambda bits: [i * 8 + (7 - k) for i, b in enumerate(bits) for k in self.byte_nonzero[b] ]

    @staticmethod
    def get_region_mask(reg_lab, regs):
        return torch.stack([(reg_lab[tuple(reg['plane_id'][:-1])].unsqueeze(-1) == torch.tensor(list(reg['ids']), device = reg_lab.device, dtype = reg_lab.dtype)).any(dim = -1) for reg in regs])

    def forward(self, *, img_bgrbhw3_255, generator = None, print = print):
        assert img_bgrbhw3_255.is_contiguous() and img_bgrbhw3_255.dtype == torch.uint8 and img_bgrbhw3_255.ndim == 4 and img_bgrbhw3_255.shape[-1] == 3
        seed = -1 if generator is None else generator.initial_seed()

        rects = torch.zeros( (img_bgrbhw3_255.shape[0], self.max_num_rects, 4), dtype = torch.int32)
        reg   = torch.zeros( (img_bgrbhw3_255.shape[0], self.max_num_rects, 5), dtype = torch.int32)
        bit   = torch.zeros( (img_bgrbhw3_255.shape[0], self.max_num_rects, self.max_num_bit), dtype = torch.uint8)
        seg   = torch.zeros( (img_bgrbhw3_255.shape[0], self.max_num_planes, img_bgrbhw3_255.shape[-3], img_bgrbhw3_255.shape[-2]), dtype = torch.int32)
        
        boxes_xywh, regions, reg_lab = [], [], []

        for k, i in enumerate(img_bgrbhw3_255):
            self.bind_num_rects.value = self.max_num_rects
            self.bind_num_planes.value = self.max_num_planes
            
            assert 0 == self.bind.process(
                i.data_ptr(), img_bgrbhw3_255.shape[-3], img_bgrbhw3_255.shape[-2], 
                rects[k].data_ptr(), ctypes.addressof(self.bind_num_rects),
                seg[k].data_ptr(), ctypes.addressof(self.bind_num_planes),
                reg[k].data_ptr(),
                bit[k].data_ptr(), bit.shape[-1],
                self.preset.encode(), self.base_k, self.inc_k, self.sigma,
                self.remove_duplicate_boxes, seed
            )
            
            boxes_xywh.append(rects[k, :self.bind_num_rects.value])
            regions.append([ dict(plane_id = (k, region_image_id, 0, 0), ids = self.bit_nonzero(b), level = region_level, id = region_id, idx = region_idx, parent_idx = region_merged_to, bbox_xywh = tuple(bbox_xywh)) for (region_id, region_level, region_image_id, region_idx, region_merged_to), bbox_xywh, b in zip( reg[k, :self.bind_num_rects.value].tolist(), rects[k, :self.bind_num_rects.value].tolist(), bit[k, :self.bind_num_rects.value].tolist() ) ])
            reg_lab.append(seg[k, :self.bind_num_planes.value])

        return boxes_xywh, regions, torch.stack(reg_lab).unsqueeze(-3)
