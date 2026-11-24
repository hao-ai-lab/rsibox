"""Cryptographic system identity: deterministic SHA-256 fingerprint of live lab state."""
import hashlib
import json
import sys

SCHEMA = "rsibox-identity/v1"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_canonical(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_fingerprint(fp):
    """Strip process-local code object ids from `{participant: {key: [hash, id]}}`."""
    if not isinstance(fp, dict):
        return {}
    return {
        str(p): ({str(k): str(v[0] if isinstance(v, (list, tuple)) and v else v)
                  for k, v in sorted(es.items())}
                 if isinstance(es, dict) else str(es))
        for p, es in sorted(fp.items())
    }


def _resolve_alias(module, alias):
    for prefix, target in (alias or {}).items():
        if module == prefix or module.startswith(prefix + "."):
            return target + module[len(prefix):]
    return module


def _fp_module_hashes(fingerprint, module):
    prefixes = (f"{module}:", f"{module}#actor:")
    out = {}
    for pname, entries in normalize_fingerprint(fingerprint).items():
        if isinstance(entries, dict):
            hits = {k: d for k, d in entries.items()
                    if (k.split("/", 1)[1] if k.startswith("rank") and "/" in k else k).startswith(prefixes)}
            if hits:
                out[pname] = hits
    return out


def module_source_digests(modules=(), sources=None, alias=None, fingerprint=None):
    """Return `{module: 64-hex sha256}` without exposing host filesystem paths."""
    src_map, digests = dict(sources or {}), {}
    for mod_name in sorted(set(modules or ()) | set(src_map)):
        src = src_map.get(mod_name)
        if isinstance(src, str):
            digests[mod_name] = sha256_text(src)
            continue
        loaded = sys.modules.get(_resolve_alias(mod_name, alias))
        mod_file = getattr(loaded, "__file__", None) if loaded is not None else None
        if isinstance(mod_file, str):
            try:
                with open(mod_file, encoding="utf-8") as fh:
                    digests[mod_name] = sha256_text(fh.read())
                continue
            except OSError:
                pass
        fp_hashes = _fp_module_hashes(fingerprint, mod_name)
        digests[mod_name] = (sha256_canonical(fp_hashes) if fp_hashes
                             else sha256_text(f"unavailable:{mod_name}"))
    return digests


def compute_identity(module_sources=None, txns=(), state="base", *, modules=(),
                     fingerprint=None, incarnation=1, recovery_required=False,
                     metadata=None, alias=None):
    """Compute deterministic SHA-256 identity dict over sources, txns, and state."""
    mod_digests = module_source_digests(modules, sources=module_sources,
                                        alias=alias, fingerprint=fingerprint)
    txn_list = [str(t) for t in (txns or ())]
    meta = dict(metadata) if isinstance(metadata, dict) else {}
    payload = {"schema": SCHEMA, "modules": mod_digests, "txns": txn_list,
               "state": str(state), "incarnation": int(incarnation),
               "recovery_required": bool(recovery_required),
               "code_hashes": normalize_fingerprint(fingerprint), "metadata": meta}
    digest = sha256_canonical(payload)
    return {"schema": SCHEMA, "digest": digest, "short": digest[:16],
            "modules": mod_digests, "txns": txn_list, "state": payload["state"],
            "incarnation": payload["incarnation"],
            "recovery_required": payload["recovery_required"], "metadata": meta}


def identity_for_receipt(record):
    """Deterministic fallback identity derived directly from a receipt record."""
    kind, txn_id = record.get("kind", "unknown"), record.get("id")
    meta = {k: record.get(k) for k in ("base_hash", "patch_hash", "engine") if k in record}
    meta.update(receipt_kind=kind, outcome=record.get("outcome") or record.get("status"))
    return compute_identity(module_sources={}, txns=[txn_id] if txn_id is not None else [],
                            state=str(meta["outcome"] or kind), metadata=meta)
