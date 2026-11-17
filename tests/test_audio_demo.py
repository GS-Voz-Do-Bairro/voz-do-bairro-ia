import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

spec = importlib.util.spec_from_file_location("avaliar_audio", Path(__file__).parent.parent / "scripts" / "avaliar_audio.py")
aa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aa)


def test_wer_identico_e_zero():
    assert aa.wer("Tem um buraco na rua", "tem um buraco na rua!") == 0.0


def test_wer_ignora_acento_e_conta_erros():
    assert aa.wer("rua das flores", "rua das flôres") == 0.0
    assert abs(aa.wer("um dois tres quatro", "um dois quatro") - 0.25) < 1e-9
    assert aa.wer("a b", "x y") == 1.0


def test_wer_referencia_vazia():
    assert aa.wer("", "") == 0.0 and aa.wer("", "algo") == 1.0


def test_demo_page_servida():
    r = TestClient(app).get("/demo")
    assert r.status_code == 200 and "MediaRecorder" in r.text
