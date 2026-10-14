"""Download published ONNX weights, keep provenance and checksum, never load pickle."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

DEFAULT_MODELS = Path('D:/fer_experiments/models/realtime')
MODELS = {
    'enet_b0_8_best_afew.onnx': {
        'url': 'https://github.com/sb-ai-lab/EmotiEffLib/raw/refs/heads/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx',
        'source': 'https://github.com/sb-ai-lab/EmotiEffLib',
        'minimum_bytes': 10_000_000,
    },
    'face_detection_yunet_2023mar.onnx': {
        'url': 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
        'source': 'https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet',
        'minimum_bytes': 100_000,
    },
}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', type=Path, default=DEFAULT_MODELS)
    args = p.parse_args()
    args.models.mkdir(parents=True, exist_ok=True)
    manifest_path = args.models / 'models.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    result = {}
    for name, spec in MODELS.items():
        path = args.models / name
        if path.exists():
            if name not in previous or digest(path) != previous[name]['sha256']:
                raise RuntimeError(f'Existing unverified/changed model: {path}; preserve it and use a different model directory.')
            result[name] = previous[name]
            print(f'Verified cached model: {name}', flush=True)
            continue
        for attempt in range(3):
            try:
                print(f'Downloading {name}, attempt {attempt + 1}', flush=True)
                request = urllib.request.Request(spec['url'], headers={'User-Agent': 'FER-Local-App/1.0'})
                temp = path.with_suffix('.onnx.part')
                with urllib.request.urlopen(request, timeout=60) as response, temp.open('wb') as f:
                    expected = response.headers.get('Content-Length')
                    total = 0
                    for chunk in iter(lambda: response.read(1024 * 1024), b''):
                        f.write(chunk)
                        total += len(chunk)
                if total < spec['minimum_bytes'] or (expected and total != int(expected)):
                    raise ValueError(f'Truncated or invalid response: {total} bytes')
                temp.replace(path)
                result[name] = {**spec, 'bytes': total, 'sha256': digest(path)}
                # Commit each success so a second-model failure remains resumable.
                manifest_path.write_text(json.dumps({**previous, **result}, indent=2), encoding='utf-8')
                print(f'OK {name}: {total:,} bytes', flush=True)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
    manifest_path.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
