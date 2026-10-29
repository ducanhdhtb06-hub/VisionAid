"""Nhịp chụp/gửi và mạng giả của ESP32 ảo (theo compile/compile.ino).

- CamBien   : thời điểm cảm biến chụp (taskCamCapture chụp liên tục, CAMERA_GRAB_LATEST).
- GrabLatest: hàng đợi khung kiểu firmware — đầy thì bỏ khung cũ, bên gửi luôn lấy khung MỚI NHẤT.
- GioiHanFps: taskCamSend — period = 1000/g_target_fps ms, chờ (period − thời gian từ lần gửi trước); fps 0 = không chờ.
- MangGia   : thời gian một lần sendBinary bị chặn (băng thông + trễ + kẹt) và khoảng rớt hẳn.
  "tot": 3 ms + băng thông 4 Mbit/s (ESP32-S3 TCP thực tế vài Mbit/s, ước lượng);
  "kem": 30 ms + rung log-normal + 1,5 Mbit/s + kẹt 300–800 ms khoảng mỗi 10 s;
  "rot": như "tot" nhưng mất hẳn trong [t_rot, t_rot + dai_rot) (giây tính từ đầu phiên).
"""
import numpy as np


class CamBien:
    def __init__(self, fps_cam: float = 15.0, rung_ms: float = 5.0, seed: int = 0):
        self.fps, self.rung = fps_cam, rung_ms / 1000.0
        self.rng = np.random.default_rng(seed)

    def thoi_diem_chup(self, t0: float, t1: float) -> list[float]:
        n = int(round((t1 - t0) * self.fps))
        return [t0 + k / self.fps + (self.rng.normal(0, self.rung) if self.rung else 0.0) for k in range(n)]


class GrabLatest:
    def __init__(self):
        self._o = None
        self.bo = 0

    def dat(self, t_chup, khung):
        """Đặt khung mới; trả khung cũ bị đè (để ghi log 'bo') hoặc None."""
        cu, self._o = self._o, (t_chup, khung)
        if cu is not None:
            self.bo += 1
        return cu

    def lay(self):
        o, self._o = self._o, None
        return o


class GioiHanFps:
    def __init__(self, fps: float):
        self.fps = fps

    def cho(self, t_bat_dau_gui_truoc: float, t_hien_tai: float) -> float:
        if not self.fps or self.fps <= 0:
            return 0.0
        chu_ky = 1.0 / self.fps
        da_qua = t_hien_tai - t_bat_dau_gui_truoc
        return round(chu_ky - da_qua, 9) if da_qua < chu_ky else 0.0


class MangGia:
    def __init__(self, ho_so: str = "tot", seed: int = 0, t_rot: float = 20.0, dai_rot: float = 3.0):
        assert ho_so in ("tot", "kem", "rot")
        self.ho_so, self.t_rot, self.dai_rot = ho_so, t_rot, dai_rot
        self.rng = np.random.default_rng(seed)
        self._ket_tiep = 10.0 * (0.5 + self.rng.random())

    def tre_gui(self, t: float, so_byte: int) -> float:
        if self.ho_so in ("tot", "rot"):
            return 0.003 + so_byte * 8 / 4e6
        tre = 0.030 + float(self.rng.lognormal(np.log(0.02), 0.6)) + so_byte * 8 / 1.5e6
        if t >= self._ket_tiep:
            tre += float(self.rng.uniform(0.3, 0.8))
            self._ket_tiep = t + 10.0 * (0.5 + self.rng.random())
        return tre

    def dang_rot(self, t: float) -> bool:
        return self.ho_so == "rot" and self.t_rot <= t < self.t_rot + self.dai_rot
