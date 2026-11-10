"""Test chạy CODE GỐC Trung Quốc (OpenAIglasses_for_Navigation.bak-zh, không sửa dòng Python nào) nói tiếng Việt:
thư mục giọng mới (VOICE_DIR) ánh xạ câu tiếng Trung -> WAV tiếng Việt theo bảng dịch lúc Việt hóa.
Chạy: .venv/bin/python -m pytest loi_do_sau/truc_tiep/test_goc_trung_quoc.py -q  (đứng ở glassai/)
"""
import json
import os
import shutil
import tempfile

import pytest

from loi_do_sau.truc_tiep.goc_trung_quoc import GOC_ZH, tao_giong_viet, tao_ban_chay

pytestmark = pytest.mark.skipif(not os.path.isdir(GOC_ZH), reason='không có bản code gốc Trung Quốc (bak-zh)')


def test_giong_viet_du_moi_cau_goc(tmp_path):
    ra = tao_giong_viet(str(tmp_path / "giong"))
    m = json.load(open(os.path.join(ra, "map.zh-CN.json"), encoding="utf-8"))
    goc = json.load(open(os.path.join(GOC_ZH, "voice", "map.zh-CN.json"), encoding="utf-8"))
    assert set(m) == set(goc)
    for k, v in m.items():
        for f in v["files"]:
            assert f.startswith("vi/") and os.path.isfile(os.path.join(ra, f)), (k, f)


def test_cau_dung_lai_thanh_tieng_viet(tmp_path):
    ra = tao_giong_viet(str(tmp_path / "giong"))
    m = json.load(open(os.path.join(ra, "map.zh-CN.json"), encoding="utf-8"))
    assert m["丢失路径，重新搜索。"]["files"] == ["vi/mat_duong_di_dang_tim_lai.wav"]


def test_ban_chay_khong_dung_vao_ban_goc():
    # liên kết cứng chỉ làm được trên cùng ổ đĩa với bản gốc (/tmp là ổ khác)
    tam = tempfile.mkdtemp(dir=os.path.dirname(GOC_ZH), prefix=".thu_ban_chay_")
    try:
        _kiem_ban_chay(os.path.join(tam, "chay"))
    finally:
        shutil.rmtree(tam)


def _kiem_ban_chay(duong):
    ra = tao_ban_chay(duong)
    # code gốc tìm model ở model/ (bak-zh để .pt ngoài thư mục gốc, model/ rỗng -> yolo_seg_model: False)
    for f in ("yolo-seg.pt", "yoloe-11l-seg.pt", "trafficlight.pt"):
        assert os.path.isfile(os.path.join(ra, "model", f)), f
    assert open(os.path.join(ra, "app_main.py"), "rb").read() == open(os.path.join(GOC_ZH, "app_main.py"), "rb").read()
    with open(os.path.join(ra, "recordings_thu.txt"), "w") as f:      # ghi vào bản chạy không sinh file ở bản gốc
        f.write("x")
    assert not os.path.exists(os.path.join(GOC_ZH, "recordings_thu.txt"))
