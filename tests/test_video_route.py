# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""B11 (2026-10-30): an analog FPV detection has a route that can CONFIRM it
— the `video` route to ATK's analog video viewer (DETECTION_DESIGN §4.2,
"The `video` route")."""
from atk_diffusion.cut import route as RT
from atk_diffusion.detect import classes as C


def test_analog_fpv_routes_to_the_video_picture_and_it_confirms():
    fpv = C.get("drone_fpv_analog")
    assert fpv.decoders == ("video",)
    assert C.DECODERS["video"]["confirms"] and fpv.confirmable
    assert C.tools_for("drone_fpv_analog")[0] == "video"
    assert "CONFIRMS" in RT.words("video")
    menu = RT.available({"class": {"cls": "drone_fpv_analog"}})
    assert menu[0] == "video" and "bench" in menu


def test_nothing_else_gained_the_route():
    others = [c.name for c in C.CLASSES if "video" in c.decoders]
    assert others == ["drone_fpv_analog"]
