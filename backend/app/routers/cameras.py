"""Cameras & zones: CRUD, start/stop monitoring, snapshot for the zone editor, live MJPEG stream."""

from __future__ import annotations

import re
import time
import uuid

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..auth import current_user
from ..config import VIDEO_DIR
from ..db import get_db
from ..models import Camera, User
from ..schemas import CameraIn, CameraOut
from ..services.vision_worker import CameraSpec, grab_snapshot, placeholder_jpeg, vision_manager

router = APIRouter(prefix="/api/cameras", tags=["cameras"], dependencies=[Depends(current_user)])


def _out(cam: Camera) -> CameraOut:
    out = CameraOut.model_validate(cam)
    out.runtime = vision_manager.runtime(cam.id)
    out.active_state = vision_manager.is_running(cam.id)
    return out


def _get(db: Session, camera_id: int) -> Camera:
    cam = db.get(Camera, camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    return cam


@router.get("", response_model=list[CameraOut])
def list_cameras(db: Session = Depends(get_db)):
    return [_out(c) for c in db.query(Camera).order_by(Camera.id).all()]


@router.post("", response_model=CameraOut, status_code=201)
def create_camera(body: CameraIn, db: Session = Depends(get_db)):
    if db.query(Camera).filter(Camera.camera_name == body.camera_name).first():
        raise HTTPException(409, "A camera with this name already exists")
    cam = Camera(**body.model_dump())
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return _out(cam)


@router.get("/{camera_id}", response_model=CameraOut)
def get_camera(camera_id: int, db: Session = Depends(get_db)):
    return _out(_get(db, camera_id))


@router.put("/{camera_id}", response_model=CameraOut)
def update_camera(camera_id: int, body: CameraIn, db: Session = Depends(get_db)):
    cam = _get(db, camera_id)
    clash = db.query(Camera).filter(Camera.camera_name == body.camera_name, Camera.id != camera_id).first()
    if clash:
        raise HTTPException(409, "A camera with this name already exists")
    for k, v in body.model_dump().items():
        setattr(cam, k, v)
    db.commit()
    if vision_manager.is_running(cam.id):  # apply new zone / thresholds immediately
        vision_manager.start(CameraSpec.from_row(cam))
    return _out(cam)


@router.delete("/{camera_id}", status_code=204)
def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    cam = _get(db, camera_id)
    vision_manager.stop(cam.id)
    db.delete(cam)
    db.commit()
    return Response(status_code=204)


@router.post("/{camera_id}/start", response_model=CameraOut)
def start_camera(camera_id: int, db: Session = Depends(get_db)):
    cam = _get(db, camera_id)
    try:
        vision_manager.start(CameraSpec.from_row(cam))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    cam.active_state = True
    db.commit()
    time.sleep(0.2)
    return _out(cam)


@router.post("/{camera_id}/stop", response_model=CameraOut)
def stop_camera(camera_id: int, db: Session = Depends(get_db)):
    cam = _get(db, camera_id)
    vision_manager.stop(cam.id)
    cam.active_state = False
    db.commit()
    return _out(cam)


@router.get("/{camera_id}/snapshot")
def snapshot(camera_id: int, db: Session = Depends(get_db)):
    """A clean frame (no overlays) so the operator can draw the restricted polygon on it."""
    cam = _get(db, camera_id)
    raw = vision_manager.raw_frame(cam.id)
    if raw is not None:
        h, w = raw.shape[:2]
        data = cv2.imencode(".jpg", raw)[1].tobytes()
    else:
        try:
            data, w, h = grab_snapshot(cam.source_type, cam.source)
        except RuntimeError as exc:
            raise HTTPException(422, str(exc)) from exc
    return Response(content=data, media_type="image/jpeg",
                    headers={"X-Frame-Width": str(w), "X-Frame-Height": str(h),
                             "Access-Control-Expose-Headers": "X-Frame-Width, X-Frame-Height",
                             "Cache-Control": "no-store"})


@router.get("/{camera_id}/stream")
def stream(camera_id: int, db: Session = Depends(get_db)):
    """multipart/x-mixed-replace MJPEG stream of the annotated frames (≈12 fps)."""
    cam = _get(db, camera_id)
    offline = placeholder_jpeg(f"{cam.camera_name}: offline - press Start")

    def gen():
        idle = 0
        while True:
            frame = vision_manager.jpeg(cam.id) or offline
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            if not vision_manager.is_running(cam.id):
                idle += 1
                if idle > 20:  # stop pushing frames for a stopped camera
                    break
            else:
                idle = 0
            time.sleep(0.08)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@router.post("/upload-video")
def upload_video(file: UploadFile = File(...), _: User = Depends(current_user)):
    """Store a CCTV-style clip locally so a camera can use it as a 'file' source."""
    if not (file.content_type or "").startswith("video/") and not re.search(
            r"\.(mp4|avi|mov|mkv|webm)$", file.filename or "", re.I):
        raise HTTPException(415, "Please upload a video file (mp4, avi, mov, mkv, webm)")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", file.filename or "clip.mp4")
    dest = VIDEO_DIR / f"{uuid.uuid4().hex[:8]}_{safe}"
    with dest.open("wb") as f:
        while chunk := file.file.read(1 << 20):
            f.write(chunk)
    return {"path": str(dest), "filename": dest.name, "size_bytes": dest.stat().st_size}


@router.get("/videos/list")
def list_videos():
    return [{"path": str(p), "filename": p.name, "size_bytes": p.stat().st_size}
            for p in sorted(VIDEO_DIR.glob("*")) if p.is_file()]
