import pytest


@pytest.fixture(autouse=True)
def _tat_thong_bao_vach(monkeypatch):
    """Test của các tính năng khác không xét 2 câu "Đã thấy / Chưa thấy vạch" (test riêng của chúng bật lại)."""
    monkeypatch.setenv("AIGLASS_THONG_BAO_VACH", "0")
