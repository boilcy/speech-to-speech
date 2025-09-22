# use camera to capture images, then run the SmolVLM-256M
# to perpare for the inference of qwen model

#!/usr/bin/env python3
import argparse, time, json, os, sys
from datetime import datetime

import cv2
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
)

# Optional Qwen import is deferred until --use-qwen is set
QWEN_AVAILABLE = False


def bgr_to_pil(frame_bgr):
    # OpenCV -> PIL (BGR -> RGB)
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def build_gst_pipeline(
    sensor_id=0, width=1280, height=720, fps=30, flip_method=0, csi=True
):
    """
    Jetson-friendly pipelines.
    - CSI camera: nvarguscamerasrc
    - USB camera: v4l2src (set csi=False and device path)
    """
    if csi:
        # CSI: nvarguscamerasrc -> NVMM -> nvvidconv -> BGR -> appsink
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1, format=NV12 ! "
            f"nvvidconv flip-method={flip_method} ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
        )
    else:
        # USB on Jetson: v4l2src
        return (
            f"v4l2src device=/dev/video{sensor_id} ! "
            f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
        )


def open_capture(args):
    if args.gstreamer:
        pipeline = build_gst_pipeline(
            sensor_id=args.cam_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
            flip_method=args.flip,
            csi=not args.usb_on_jetson_is_v4l2,
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print(
                "[WARN] GStreamer pipeline failed; falling back to cv2.VideoCapture(index)."
            )
            cap = cv2.VideoCapture(args.cam_index)
    else:
        cap = cv2.VideoCapture(args.cam_index)
        # Try to set basic properties if supported
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    return cap


def load_smolvlm(device, dtype, model_id):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        # flash-attn 2 speeds up on desktop GPUs; on Jetson keep 'eager'
        _attn_implementation="flash_attention_2"
        if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8)
        else "eager",
    ).to(device)
    return processor, model


def caption_with_smolvlm(
    proc, model, pil_img, prompt_text="Describe this image briefly."
):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }
    ]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=prompt, images=[pil_img], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=64)
    text = proc.batch_decode(out_ids, skip_special_tokens=True)[0]
    # Strip a common "Assistant: " prefix if present
    return text.split("Assistant:", 1)[-1].strip()


def maybe_load_qwen(device, dtype, qwen_id):
    global QWEN_AVAILABLE
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # noqa: F401

    QWEN_AVAILABLE = True
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        qwen_id,
        torch_dtype=dtype,
        device_map="auto",
        # use FA2 if available
        attn_implementation="flash_attention_2"
        if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8)
        else "eager",
    )
    proc = __import__("transformers").Qwen2_5OmniProcessor.from_pretrained(qwen_id)
    return proc, model


def ask_qwen(
    qproc,
    qmodel,
    pil_img,
    smol_caption,
    user_question="What key facts should I know about the scene?",
):
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a precise vision assistant."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {
                    "type": "text",
                    "text": f"Auto caption: {smol_caption}\n\n{user_question} Keep it short.",
                },
            ],
        },
    ]
    inputs = qproc.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    ).to(qmodel.device)
    with torch.no_grad():
        out_ids = qmodel.generate(**inputs, max_new_tokens=96)
    text = qproc.batch_decode(out_ids, skip_special_tokens=True)[0]
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Camera -> SmolVLM-256M -> (optional) Qwen2.5-Omni"
    )
    parser.add_argument(
        "--cam-index", type=int, default=0, help="Camera index or sensor-id"
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--gstreamer",
        action="store_true",
        help="Use GStreamer pipeline (Jetson friendly)",
    )
    parser.add_argument(
        "--usb-on-jetson-is-v4l2",
        action="store_true",
        help="Use v4l2src pipeline instead of nvarguscamerasrc",
    )
    parser.add_argument(
        "--flip", type=int, default=0, help="nvvidconv flip-method for CSI (0..7)"
    )

    parser.add_argument(
        "--model-smolvlm", default="HuggingFaceTB/SmolVLM-256M-Instruct"
    )
    parser.add_argument("--model-qwen", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument(
        "--use-qwen", action="store_true", help="Also query Qwen with image+caption"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0, help="Seconds between inferences"
    )
    parser.add_argument(
        "--save", action="store_true", help="Save frames and JSONL logs"
    )
    parser.add_argument("--out-dir", default="captures")
    parser.add_argument(
        "--overlay", action="store_true", help="Draw caption on the preview window"
    )
    parser.add_argument(
        "--question", default="What key facts should I know about the scene?"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Jetson often lacks bfloat16 — prefer fp16 on CUDA; otherwise float32
    dtype = torch.float16 if device == "cuda" else torch.float32

    if args.save:
        os.makedirs(args.out_dir, exist_ok=True)
        log_path = os.path.join(args.out_dir, "log.jsonl")
        log_f = open(log_path, "a", encoding="utf-8")
    else:
        log_f = None

    cap = open_capture(args)

    # Load SmolVLM
    print(f"[INFO] Loading SmolVLM on {device}...")
    sproc, smodel = load_smolvlm(device, dtype, args.model_smolvlm)

    # Load Qwen only if requested
    if args.use_qwen:
        print(f"[INFO] Loading Qwen on {device}...")
        try:
            qproc, qmodel = maybe_load_qwen(device, dtype, args.model_qwen)
        except Exception as e:
            print(f"[WARN] Could not load Qwen: {e}")
            args.use_qwen = False

    last = 0.0
    print("[INFO] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame.")
            time.sleep(0.05)
            continue

        now = time.time()
        show = frame.copy()

        if now - last >= args.interval:
            last = now
            pil_img = bgr_to_pil(frame)
            t0 = time.time()
            caption = caption_with_smolvlm(sproc, smodel, pil_img)
            t1 = time.time()

            qwen_reply = None
            if args.use_qwen and QWEN_AVAILABLE:
                qwen_reply = ask_qwen(qproc, qmodel, pil_img, caption, args.question)

            # Console out
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{ts}] Caption ({t1 - t0:.2f}s): {caption}")
            if qwen_reply:
                print(f"[Qwen] {qwen_reply}")

            # Overlay for preview
            if args.overlay:
                txt = (caption[:80] + "…") if len(caption) > 80 else caption
                cv2.rectangle(show, (0, 0), (show.shape[1], 40), (0, 0, 0), -1)
                cv2.putText(
                    show,
                    txt,
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Save
            if args.save:
                fname = f"{int(now)}.jpg"
                img_path = os.path.join(args.out_dir, fname)
                cv2.imwrite(img_path, frame)
                rec = {"ts": ts, "image": img_path, "caption": caption}
                if qwen_reply:
                    rec["qwen"] = qwen_reply
                log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log_f.flush()

        # Preview
        cv2.imshow("Live (SmolVLM -> Qwen)", show)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if log_f:
        log_f.close()


if __name__ == "__main__":
    main()
