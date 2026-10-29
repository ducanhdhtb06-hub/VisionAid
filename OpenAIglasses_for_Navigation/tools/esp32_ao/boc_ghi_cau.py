"""Chạy app_main.py của một bản code KHÔNG có nhật ký câu (code gốc) mà vẫn ghi được câu nói, không sửa dòng nào:
bọc audio_player.play_voice_text TRƯỚC khi app_main import nó, mỗi lần gọi ghi một dòng JSON vào AIGLASS_NHAT_KY_CAU
{t, cau, phat, ly_do} như audio_player bản mới. phat = audio_player._last_voice_time đổi sau lần gọi (gốc chỉ cập nhật
khi thật sự phát; trùng câu trong 1 s thì không).
Dùng: (đứng ở thư mục bản code) python boc_ghi_cau.py  — phat_lai.bat_may_chu(..., boc_ghi_cau=True) tự làm.
"""
import json
import os
import runpy
import sys
import time

sys.path.insert(0, os.getcwd())
import audio_player as A  # noqa: E402

_goc = A.play_voice_text
_duong = os.environ["AIGLASS_NHAT_KY_CAU"]


def _boc(text):
    truoc = A._last_voice_time
    _goc(text)
    if text:
        with open(_duong, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(t=time.time(), cau=text, phat=A._last_voice_time != truoc,
                                    ly_do="" if A._last_voice_time != truoc else "khong_phat"),
                               ensure_ascii=False) + "\n")


A.play_voice_text = _boc
sys.argv = ["app_main.py"]
runpy.run_path("app_main.py", run_name="__main__")
