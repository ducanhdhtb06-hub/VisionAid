"""Chạy CODE GỐC Trung Quốc y hệt (OpenAIglasses_for_Navigation.bak-zh, bản sao lưu 22/10 trước Việt hóa) nhưng nói
tiếng Việt — để so cách dẫn đường gốc với bản mới qua iPhone.

Không sửa dòng Python nào của bản gốc:
- giọng: audio_player gốc đọc VOICE_DIR/map.zh-CN.json -> tạo thư mục giọng mới, mỗi câu tiếng Trung trỏ tới WAV
  tiếng Việt theo bảng dịch lúc Việt hóa (i18n/voice_vi.json, phủ 118/118 câu);
- chạy trong bản sao bằng liên kết cứng (không tốn đĩa, không ghi gì vào bản sao lưu).
Chữ trên giao diện/log của bản gốc vẫn là tiếng Trung; lệnh bật/tắt phải là "开始导航"/"停止导航".
"""
import json
import os
import shutil

GLASSAI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOC_ZH = os.path.join(GLASSAI, "OpenAIglasses_for_Navigation.bak-zh")
MOI = os.path.join(GLASSAI, "OpenAIglasses_for_Navigation")
CAU_BAT_ZH, CAU_TAT_ZH = "开始导航", "停止导航"


def tao_giong_viet(ra: str) -> str:
    with open(os.path.join(GOC_ZH, "voice", "map.zh-CN.json"), encoding="utf-8") as f:
        goc = json.load(f)
    with open(os.path.join(MOI, "i18n", "voice_vi.json"), encoding="utf-8") as f:
        dich = json.load(f)
    with open(os.path.join(MOI, "voice", "map.vi.json"), encoding="utf-8") as f:
        vi = json.load(f)
    os.makedirs(ra, exist_ok=True)
    lk = os.path.join(ra, "vi")
    if not os.path.exists(lk):
        os.symlink(os.path.join(MOI, "voice", "vi"), lk)
    m = {k: dict(v, files=vi[dich[k]]["files"]) for k, v in goc.items()}
    with open(os.path.join(ra, "map.zh-CN.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    return ra


def tao_ban_chay(ra: str) -> str:
    if not os.path.exists(ra):
        shutil.copytree(GOC_ZH, ra, copy_function=os.link, ignore=shutil.ignore_patterns("__pycache__", "recordings"))
    # code gốc đọc model/<tên>.pt; bản sao lưu để .pt ở thư mục gốc (cùng md5 với model/ của bản mới)
    os.makedirs(os.path.join(ra, "model"), exist_ok=True)
    for f in os.listdir(ra):
        if f.endswith(".pt") and not os.path.exists(os.path.join(ra, "model", f)):
            os.link(os.path.join(ra, f), os.path.join(ra, "model", f))
    clip = os.path.join(MOI, "mobileclip_blt.ts")          # YOLOE cần; thiếu thì ultralytics tự tải qua mạng
    if os.path.exists(clip) and not os.path.exists(os.path.join(ra, "mobileclip_blt.ts")):
        os.link(clip, os.path.join(ra, "mobileclip_blt.ts"))
    return ra
