"""Mede o WER (taxa de erro de palavras) da transcrição com ÁUDIOS REAIS gravados pela equipe.

Uso: python scripts/avaliar_audio.py data/audios/   (OPENAI_API_KEY definida)
Cada áudio `xxx.m4a|mp3|wav|webm` precisa de `xxx.txt` com o que foi realmente dito (referência).
"""
import re
import sys
import unicodedata
from pathlib import Path

EXT = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".mp4", ".mpeg", ".mpga", ".flac"}


def normalizar(t: str) -> list[str]:
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", t).split()


def wer(ref: str, hip: str) -> float:
    r, h = normalizar(ref), normalizar(hip)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i, pr in enumerate(r, 1):
        ant, d[0] = d[0], i
        for j, ph in enumerate(h, 1):
            ant, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, ant + (pr != ph))
    return d[len(h)] / len(r)


def main(pasta: str) -> None:
    from vozia.transcribe import Transcritor
    tr = Transcritor()
    linhas, tot = [], []
    for a in sorted(Path(pasta).iterdir()):
        ref = a.with_suffix(".txt")
        if a.suffix.lower() in EXT and ref.exists():
            hip = tr.transcrever(a.read_bytes(), a.name)
            w = wer(ref.read_text(encoding="utf-8"), hip)
            tot.append(w)
            linhas.append(f"| {a.name} | {w:.2%} | {hip[:70]} |")
    if not tot:
        raise SystemExit("nenhum par áudio + .txt encontrado")
    out = ["| Áudio | WER | Transcrição |", "|---|---|---|"] + linhas + ["", f"**WER médio: {sum(tot)/len(tot):.2%}** em {len(tot)} áudios"]
    Path("reports/wer.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main(sys.argv[1] if len(sys.argv) > 1 else "data/audios")
