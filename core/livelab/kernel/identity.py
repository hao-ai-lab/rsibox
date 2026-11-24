"""Cryptographic system identity: deterministic SHA-256 fingerprint of live lab state."""
import hashlib
import json
import sys

SCHEMA = "rsibox-identity/v1"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_canonical(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_fingerprint(fp):
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
    return {
        pname: hits
        for pname, entries in normalize_fingerprint(fingerprint).items()
        if isinstance(entries, dict) and (hits := {
            k: d for k, d in entries.items()
            if (k.split("/", 1)[1] if k.startswith("rank") and "/" in k else k).startswith(prefixes)
        })
    }


def module_source_digests(modules=(), sources=None, alias=None, fingerprint=None):
    """Return `{module: 64-hex sha256}`, preferring in-memory sources and live code hashes."""
    src_map, digests = dict(sources or {}), {}
    for mod_name in sorted(set(modules or ()) | set(src_map)):
        if isinstance(src_map.get(mod_name), str):
            digests[mod_name] = sha256_text(src_map[mod_name])
        elif (fp_hashes := _fp_module_hashes(fingerprint, mod_name)):
            digests[mod_name] = sha256_canonical(fp_hashes)
        else:
            loaded = sys.modules.get(_resolve_alias(mod_name, alias))
            mod_file = getattr(loaded, "__file__", None) if loaded is not None else None
            try:
                if not isinstance(mod_file, str):
                    raise OSError
                with open(mod_file, encoding="utf-8") as fh:
                    digests[mod_name] = sha256_text(fh.read())
            except (OSError, UnicodeDecodeError):
                digests[mod_name] = sha256_text(f"unavailable:{mod_name}")
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
