import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import to_rgb

def string_to_rgb(color_string):
    color_rgb = to_rgb(color_string)
    return tuple(int(c * 255) for c in color_rgb)


def repeat_tensors(tensor, repeat_counts):
    repeated_tensors = [tensor[i:i+1].repeat(repeat, *[1] * (tensor.ndim - 1)) for i, repeat in enumerate(repeat_counts)]
    return torch.cat(repeated_tensors, dim=0)

def split_tensors(tensor, split_counts):
    indices = torch.cumsum(torch.tensor([0] + split_counts), dim=0)
    return [tensor[indices[i]:indices[i+1]] for i in range(len(split_counts))]

def stack_and_pad(tensor_list):
    max_size = max([t.shape[0] for t in tensor_list])
    padded_list = []
    for t in tensor_list:
        if t.shape[0] == max_size:
            padded_list.append(t)
        else:
            padded_list.append(torch.cat([t, torch.zeros(max_size - t.shape[0], *t.shape[1:])], dim=0))
    return torch.stack(padded_list)


def heatmap_gaze(blank_image, heatmap):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    size = blank_image.shape[:2][::-1]
    heatmap = cv2.resize((heatmap * 255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)     
    overlay_image = cv2.addWeighted(blank_image, 0.5, heatmap, 0.5, 0)
    return overlay_image

def draw_gaze_lines(image, results):
    overlay_image = image.copy()
    colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow']
    width, height = image.shape[1], image.shape[0]
    for key, data in results.items():
        face = data['face']
        xmin, ymin, xmax, ymax = face
        color = string_to_rgb(colors[key % len(colors)])
        cv2.rectangle(overlay_image, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, 2)
        if data['heatmap'] is not None:
            heatmap_np = data['heatmap']
            heatmap_image = heatmap_gaze(image, heatmap_np)
            lines = data['line']['start'], data['line']['end']
            cv2.line(overlay_image, (int(lines[0][0]), int(lines[0][1])), (int(lines[1][0]), int(lines[1][1])), color, 2)
    return overlay_image, heatmap_image

def draw_gaze_arrow(image, result):
    overlay_image = image.copy()
    color = (255, 255, 0)
    face = result['face']
    xmin, ymin, xmax, ymax = face
    width, height = image.shape[1], image.shape[0]
    cv2.rectangle(overlay_image, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, 2)
    lines = result['line']['start'], result['line']['end']
    heatmap_image = heatmap_gaze(image, result['heatmap'])
    cv2.line(overlay_image, (int(lines[0][0]), int(lines[0][1])), (int(lines[1][0]), int(lines[1][1])), color, 2)
    return overlay_image, heatmap_image