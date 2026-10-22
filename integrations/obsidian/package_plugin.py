"""Reproducible, local-only Obsidian package builder and integrity checker.

Never installs a plugin, modifies a vault or starts Grimore. All inputs are
explicitly allowlisted source files; all output is in a newly created folder.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import stat
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ID = "grimore-readonly"
RUNTIME_LIMITS = {"main.js": 512 * 1024, "manifest.json": 16 * 1024, "styles.css": 64 * 1024}
LIMITS = {**RUNTIME_LIMITS, "LICENSE": 16 * 1024}
MAX_BUNDLE_BYTES = sum(LIMITS.values()) + 4096
MAX_RELEASE_BYTES = 32 * 1024
VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
INSTALL_TEXT = """Grimore para Obsidian — entrega local pendiente de aceptación

Este paquete no ha sido instalado ni aceptado en un Obsidian real.
No lo consideres una publicación del catálogo comunitario.

1. Verifica el ZIP con package_plugin.py verify y el repositorio de confianza.
   Una suma SHA-256 aislada no autentica al autor ni certifica la seguridad.
2. Extrae el ZIP en una carpeta temporal, nunca encima de una instalación.
3. Conserva una copia de seguridad de la bóveda y de cualquier complemento
   existente con el mismo ID antes de sustituirlo manualmente.
4. Copia grimore-readonly (tres archivos de ejecución y LICENSE) al directorio
   de complementos de tu bóveda de prueba. No copies release.json ni bases
   de datos. No se modifica el modo restringido automáticamente.
5. Habilita el complemento y configura sus rutas. No se conecta al cargar.
   La configuración de Grimore debe apuntar a la misma bóveda física.
6. Prueba búsqueda, fuentes, alertas, apertura y desconexión.

Solo escritorio. Sin generación de respuestas ni edición de notas/decisiones.
Los embeddings remotos requieren autorización expresa. Grimore puede mantener
sus cachés normales. Un timeout no garantiza cancelar un modelo remoto.

Para retirar: desconecta, deshabilita y guarda tus ajustes antes de quitar
manualmente la carpeta. No hay cambios de notas que restaurar por este plugin.
Las pruebas de aceptación y el piloto de dos semanas siguen pendientes.
"""


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _json_object(data: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON contains duplicate keys")
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError("JSON must contain finite values")

    try:
        result = json.loads(data, object_pairs_hook=unique, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid JSON document") from exc
    if not isinstance(result, dict):
        raise ValueError("JSON must be an object")
    return result


def _read_file(path: Path, limit: int) -> bytes:
    # Do not follow links into installed settings or unrelated private files.
    if path.is_symlink() or path.absolute() != path.resolve() or not path.is_file():
        raise ValueError("Source/package must be a regular file without symlink ancestors")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if not data or len(data) > limit:
        raise ValueError("Source/package is empty or exceeds its size limit")
    return data


def prepare(repo_root: Path = REPO_ROOT) -> tuple[dict, dict[str, bytes]]:
    """Validate a source-only directory without any writes or model calls."""
    root = repo_root.absolute()
    source = root / "integrations" / "obsidian" / PLUGIN_ID
    if source.is_symlink() or source != source.resolve() or not source.is_dir():
        raise ValueError("Plugin source must be a regular directory without symlink ancestors")
    if {entry.name for entry in source.iterdir()} != set(RUNTIME_LIMITS):
        raise ValueError("Plugin source must contain only main.js, manifest.json and styles.css")
    files = {name: _read_file(source / name, limit) for name, limit in RUNTIME_LIMITS.items()}
    manifest = _json_object(files["manifest.json"])
    if (manifest.get("id") != PLUGIN_ID or manifest.get("isDesktopOnly") is not True
            or not isinstance(manifest.get("version"), str)
            or VERSION.fullmatch(manifest["version"]) is None
            or not isinstance(manifest.get("minAppVersion"), str)
            or VERSION.fullmatch(manifest["minAppVersion"]) is None):
        raise ValueError("Plugin manifest has an unsupported identity, version or desktop policy")
    for key in ("name", "description", "author"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError("Plugin manifest is missing required text metadata")
    for name, data in files.items():
        try:
            data.decode("utf-8")
        except UnicodeError as exc:
            raise ValueError(f"Runtime source is not UTF-8: {name}") from exc
    files["LICENSE"] = _read_file(root / "LICENSE", LIMITS["LICENSE"])
    return manifest, files


def _archive(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(f"{PLUGIN_ID}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, data)
    return stream.getvalue()


def _release(manifest: dict, files: dict[str, bytes], bundle: bytes) -> dict:
    return {
        "format_version": 1,
        "plugin_id": PLUGIN_ID,
        "plugin_version": manifest["version"],
        "acceptance": "pending_user",
        "installed": False,
        "artifact": {
            "name": f"{PLUGIN_ID}-{manifest['version']}.zip",
            "size_bytes": len(bundle), "sha256": _digest(bundle),
        },
        "files": [
            {"path": f"{PLUGIN_ID}/{name}", "size_bytes": len(data), "sha256": _digest(data)}
            for name, data in sorted(files.items())
        ],
    }


def verify(bundle_path: Path, release_path: Path, *, repo_root: Path = REPO_ROOT) -> dict:
    """Read a bounded ZIP without extracting it; compare with trusted sources.

    Checksums prove identity/integrity, not authorship, safety or acceptance.
    Recomputing a tampered manifest alone cannot bypass source comparison.
    """
    manifest, sources = prepare(repo_root)
    bundle = _read_file(bundle_path.absolute(), MAX_BUNDLE_BYTES)
    release_bytes = _read_file(release_path.absolute(), MAX_RELEASE_BYTES)
    release = _json_object(release_bytes)
    expected = _release(manifest, sources, bundle)
    if (release != expected or release_bytes != _json_bytes(expected)
            or bundle_path.name != expected["artifact"]["name"]):
        raise ValueError("Release record does not match the bundle and current source metadata")
    expected_paths = {f"{PLUGIN_ID}/{name}" for name in sources}
    try:
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            entries = archive.infolist()
            if len(entries) != len(expected_paths) or {info.filename for info in entries} != expected_paths:
                raise ValueError("Bundle contains missing, duplicate or unexpected paths")
            for info in entries:
                name = info.filename.removeprefix(PLUGIN_ID + "/")
                if (info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1
                        or info.file_size > LIMITS[name] or info.compress_size != info.file_size
                        or stat.S_IFMT(info.external_attr >> 16) != stat.S_IFREG):
                    raise ValueError("Bundle entry has unsupported size, compression or file type")
                with archive.open(info) as stream:
                    data = stream.read(LIMITS[name] + 1)
                if data != sources[name]:
                    raise ValueError("Bundle entry differs from the current trusted source")
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise ValueError("Invalid or unsupported ZIP bundle") from exc
    if bundle != _archive(sources):
        raise ValueError("Bundle is not the canonical reproducible artifact")
    return {"ok": True, "integrity": "verified_against_current_source",
            "plugin_version": manifest["version"], "sha256": _digest(bundle),
            "installed": False, "acceptance": "pending_user"}


def build(output_dir: Path, *, repo_root: Path = REPO_ROOT) -> dict:
    manifest, files = prepare(repo_root)
    bundle = _archive(files)
    release = _release(manifest, files, bundle)
    # Exclusive directory creation: existing releases, installations and
    # even empty directories are never overwritten. Parent must already exist.
    output_dir = output_dir.absolute()
    if output_dir.parent != output_dir.parent.resolve():
        raise ValueError("Output parent must be a canonical directory without symlink aliases")
    output_dir.mkdir(exist_ok=False)
    outputs = {
        release["artifact"]["name"]: bundle,
        "release.json": _json_bytes(release),
        "SHA256SUMS": (
            f"{_digest(bundle)}  {release['artifact']['name']}\n"
            f"{_digest(_json_bytes(release))}  release.json\n"
        ).encode("ascii"),
        "INSTALL_ES.txt": INSTALL_TEXT.encode("utf-8"),
    }
    for name, data in outputs.items():
        with (output_dir / name).open("xb") as stream:
            stream.write(data)
    result = verify(output_dir / release["artifact"]["name"], output_dir / "release.json", repo_root=repo_root)
    return {**result, "output_dir": str(output_dir), "artifact": release["artifact"]["name"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Validate allowlisted sources; no writes")
    builder = sub.add_parser("build", help="Create a new release folder; never installs or overwrites")
    builder.add_argument("--output-dir", required=True, type=Path)
    verifier = sub.add_parser("verify", help="Read-only integrity check against current repository sources")
    verifier.add_argument("--bundle", required=True, type=Path)
    verifier.add_argument("--release-json", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            manifest, files = prepare()
            result = {"ok": True, "plugin_version": manifest["version"],
                      "files": sorted(files), "writes": False, "installed": False,
                      "acceptance": "pending_user"}
        elif args.command == "build":
            result = build(args.output_dir)
        else:
            result = verify(args.bundle, args.release_json)
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc), "installed": False}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
