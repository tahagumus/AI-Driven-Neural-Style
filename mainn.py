


import cv2
import argparse
import os
import time
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def frame_to_tensor(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = transforms.ToTensor()
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    return tensor

def tensor_to_frame(tensor):
    image = tensor.squeeze(0).cpu().clamp(0, 1)
    image = transforms.ToPILImage()(image)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return frame

from style_transfer import stylize, load_image
from temporal_consistency import TemporalSmoother


def process_frame(frame, style_img, prev, args):
    content = frame
    result = stylize(content, style_img)
    return result, result


def run_webcam(style_img, args):
    cap = cv2.VideoCapture(args.camera)
    smoother = TemporalSmoother(alpha=0.7)
    prev_tensor = None
    fps = 0.0

    print(f"Cihaz: {DEVICE}  |  Adım: {args.steps}  |  Q=çıkış")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        styled, prev_tensor = process_frame(frame, style_img, prev_tensor, args)
        fps = 0.9 * fps + 0.1 / max(time.time() - t0, 1e-3)

        styled = cv2.resize(styled, (frame.shape[1], frame.shape[0]))

        if args.temporal:
            
            styled = smoother.apply(frame, styled)

        # Basit bilgi çubuğu
        display = cv2.resize(styled, frame.shape[1::-1])
        cv2.putText(display, f"FPS:{fps:.1f}  sw:{args.style_weight:.0e}  [T]temporal={'on' if args.temporal else 'off'}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,100), 1)
        cv2.imshow("Neural Style", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            os.makedirs("output", exist_ok=True)
            name = f"output/frame_{int(time.time())}.jpg"
            cv2.imwrite(name, display)
            print(f"Kaydedildi: {name}")
        elif key == ord('t'):
            args.temporal = not args.temporal
            smoother.reset()
            prev_tensor = None
        elif key == ord('+'):
            args.style_weight = min(args.style_weight * 2, 1e9)
            prev_tensor = None
        elif key == ord('-'):
            args.style_weight = max(args.style_weight / 2, 1e3)
            prev_tensor = None

    cap.release()
    cv2.destroyAllWindows()


def run_video(style_img, args):
    cap    = cv2.VideoCapture(args.video)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    w, h   = int(cap.get(3)), int(cap.get(4))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs("output", exist_ok=True)
    out = cv2.VideoWriter("output/result.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    smoother    = TemporalSmoother()
    prev_tensor = None
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        styled, prev_tensor = process_frame(frame, style_img, prev_tensor, args)
        
        styled = cv2.resize(styled, (frame.shape[1], frame.shape[0]))
        
        if args.temporal:
            styled = smoother.apply(frame, styled)
        out.write(cv2.resize(styled, (w, h)))
        i += 1
        print(f"\r{i}/{total}", end="", flush=True)

    cap.release()
    out.release()
    print("\nKaydedildi: output/result.mp4")


def run_image(style_img, args):
    frame  = cv2.imread(args.image)
    styled, _ = process_frame(frame, style_img, None, args)
    styled = cv2.resize(styled, frame.shape[1::-1])
    cv2.imwrite("output/result.jpg", styled)
    cv2.imshow("Sonuç (herhangi bir tuş)", np.hstack([frame, styled]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--style",        required=True)
    p.add_argument("--video",        default=None)
    p.add_argument("--image",        default=None)
    p.add_argument("--camera",       type=int,   default=0)
    p.add_argument("--steps",        type=int,   default=100)
    p.add_argument("--style-weight", type=float, default=1e6, dest="style_weight")
    p.add_argument("--temporal",     action="store_true", default=True)
    p.add_argument("--no-temporal",  action="store_false", dest="temporal")
    args = p.parse_args()

    style_img = load_image(args.style)

    if   args.video: run_video(style_img, args)
    elif args.image: run_image(style_img, args)
    else:            run_webcam(style_img, args)
