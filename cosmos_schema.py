"""
Read a vLLM-Omni server's /openapi.json and report what it can actually do.

Cosmos 3 advertises six input/output modality combinations (text→image,
text|video→video, text|image→video, →text for VLM reasoning, action-conditioned
video, and video|text→video|action). Any single vLLM-Omni build serves only
some of them, under field names that are documented nowhere except this schema,
and a guessed field name comes back as an opaque 422. So: ask the server.

Used two ways.

As a command line tool, over SSH, before the GUI is even up:

    python cosmos_schema.py                        # coverage report + endpoints
    python cosmos_schema.py POST /v1/videos/sync   # fields and a runnable snippet
    python cosmos_schema.py --save /data           # keep a snapshot on disk

As a library, which is what the GUI's API schema tab does:

    import cosmos_schema
    report = cosmos_schema.overview()
    info, rows, snippet, raw = cosmos_schema.describe_operation("POST /v1/videos/sync")

Nothing here imports gradio, so it can be tested and run on its own.

Environment:
  COSMOS_SERVER   vLLM-Omni endpoint (default http://localhost:8000)
"""

import json
import os
import re
import sys
import time

import requests

# COSMOS_SERVER is the single source of truth and the GUI reads the same
# variable, so the two cannot drift apart at runtime.
SERVER = os.environ.get("COSMOS_SERVER", "http://localhost:8000")

IMAGE_PATH = "/v1/images/generations"
VIDEO_PATH = "/v1/videos/sync"


# ---------------------------------------------------------------------------
# Server introspection.
#
# Cosmos 3 advertises six input/output modality combinations, but a given
# vLLM-Omni build only serves some of them, under field names that are not
# documented anywhere outside this schema. A guessed field name comes back as
# an opaque 422, so the schema is read from the server and kept visible in the
# UI rather than curl'd once and forgotten.
# ---------------------------------------------------------------------------

SCHEMA = {"spec": None, "error": "", "fetched": None}

# Endpoints beyond the two this GUI already drives, looked for by name so the
# report can say which of the remaining modalities are even reachable.
CHAT_PATHS = ("/v1/chat/completions", "/v1/responses")
ACTION_WORDS = ("action",)
ROBOT_WORDS = ("joint", "eef", "end_effector", "trajectory", "state", "control", "embodiment")


def fetch_spec(force=False):
    """Read and cache /openapi.json. Returns (spec, error_text).

    Cached because three callers want it — the connection banner, the endpoint
    list and the detail view — and they must not disagree about what the server
    looks like mid-session.
    """
    if SCHEMA["spec"] is not None and not force:
        return SCHEMA["spec"], ""
    try:
        r = requests.get(f"{SERVER}/openapi.json", timeout=10)
        r.raise_for_status()
        spec = r.json()
    except Exception as e:
        SCHEMA.update(spec=None, error=f"Could not read {SERVER}/openapi.json ({e})")
        return None, SCHEMA["error"]
    if not isinstance(spec, dict) or not isinstance(spec.get("paths"), dict):
        SCHEMA.update(spec=None,
                      error=f"{SERVER}/openapi.json answered, but not with an OpenAPI document.")
        return None, SCHEMA["error"]
    SCHEMA.update(spec=spec, error="", fetched=time.strftime("%Y-%m-%d %H:%M:%S"))
    return spec, ""


def server_paths():
    """Endpoint paths this server actually exposes, or an empty set if we
    cannot ask. Cosmos3-Nano is a video model and some builds do not serve an
    image endpoint at all, so the Image tab cannot be assumed to work."""
    # force: this runs from the connection check, where a cached answer would
    # defeat the point of checking.
    spec, _ = fetch_spec(force=True)
    return set(spec.get("paths", {})) if spec else set()


def deref(node, spec, seen=()):
    """Resolve $ref pointers in place so a schema can be read in one screen.

    vLLM puts every request body in #/components/schemas, so the raw operation
    is a wall of refs that tells you nothing. Schemas can contain themselves,
    hence the cycle guard.
    """
    if isinstance(node, list):
        return [deref(i, spec, seen) for i in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str):
        if ref in seen:
            return {"type": f"(recursive → {ref.rsplit('/', 1)[-1]})"}
        if not ref.startswith("#/"):
            return {"type": f"(external → {ref})"}
        target = spec
        for part in ref[2:].split("/"):
            if not isinstance(target, dict) or part not in target:
                return {"type": f"(unresolved → {ref})"}
            target = target[part]
        return deref(target, spec, seen + (ref,))
    return {k: deref(v, spec, seen) for k, v in node.items()}


def type_name(s):
    """A short human-readable type for one schema node."""
    if not isinstance(s, dict):
        return "?"
    if s.get("format") == "binary":
        return "file (binary)"
    t = s.get("type")
    if t == "array":
        return f"array[{type_name(s.get('items') or {})}]"
    if isinstance(t, str):
        return f"{t} ({s['format']})" if s.get("format") else t
    for key in ("anyOf", "oneOf", "allOf"):
        members = s.get(key)
        if isinstance(members, list):
            names = [type_name(m) for m in members
                     if not (isinstance(m, dict) and m.get("type") == "null")]
            # dict.fromkeys keeps declaration order while dropping duplicates,
            # which anyOf lists are full of.
            joined = " | ".join(dict.fromkeys(names))
            return joined or "null"
    if s.get("enum"):
        return "enum"
    return "object" if s.get("properties") else "?"


def is_scalar(typ):
    """Whether type_name() described something that fits in one form field."""
    head = typ.split("|")[0].split("(")[0].strip()
    return head in ("string", "integer", "number", "boolean", "enum")


def field_rows(schema):
    """One row per property of a request body: name, type, required, default, notes."""
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict):
        return []
    required = set(schema.get("required") or [])
    rows = []
    for name, s in props.items():
        s = s if isinstance(s, dict) else {}
        notes = []
        if s.get("enum"):
            notes.append("one of: " + ", ".join(str(v) for v in s["enum"][:8]))
        for k in ("minimum", "maximum", "contentMediaType"):
            if k in s:
                notes.append(f"{k}: {s[k]}")
        if s.get("description"):
            notes.append(" ".join(str(s["description"]).split()))
        default = s.get("default")
        rows.append([
            name,
            type_name(s),
            "yes" if name in required else "",
            "" if default is None else str(default),
            "; ".join(notes)[:400],
        ])
    # Required fields first: those are the ones that decide whether a new tab
    # is even possible.
    rows.sort(key=lambda r: (r[2] != "yes", r[0]))
    return rows


def body_schema(op):
    """(content_type, schema) for the request body worth showing.

    multipart wins when both are offered, because that is the harder one to
    build by hand and the one the video endpoint actually uses.
    """
    content = (op.get("requestBody") or {}).get("content")
    if not isinstance(content, dict) or not content:
        return None, {}
    for preferred in ("multipart/form-data", "application/json"):
        if preferred in content:
            return preferred, (content[preferred] or {}).get("schema") or {}
    ctype = next(iter(content))
    return ctype, (content[ctype] or {}).get("schema") or {}


def operation_choices(spec):
    """(label, key) pairs for every operation, video and image endpoints first."""
    ops = []
    for path, item in (spec.get("paths") or {}).items():
        if not isinstance(item, dict):
            continue
        for method, op in item.items():
            if method.lower() not in ("get", "post", "put", "patch", "delete"):
                continue
            summary = " ".join(str((op or {}).get("summary") or "").split())
            label = f"{method.upper():<6} {path}"
            if summary:
                label += f"  — {summary}"
            ops.append((label[:120], f"{method.upper()} {path}"))
    known = {VIDEO_PATH: 0, IMAGE_PATH: 1}
    ops.sort(key=lambda p: (known.get(p[1].split(" ", 1)[1], 2), p[1]))
    return ops


def request_snippet(method, path, ctype, rows):
    """A runnable requests call for this endpoint.

    The point is that a new tab starts from something that already talks to
    this server correctly, instead of from a guess about field names.
    """
    if method != "POST" or not rows:
        return (f'r = requests.{method.lower()}(f"{{SERVER}}{path}", timeout=30)\n'
                f'print(r.status_code, r.text[:500])')

    binaries = [r for r in rows if "binary" in r[1]]
    plain = [r for r in rows if "binary" not in r[1]]
    lines = []

    if ctype == "multipart/form-data":
        # Every value is a string here: multipart has no types, and FastAPI
        # parses "true"/"false" and "24" back out on the other side.
        lines.append("form = {")
        for name, typ, req, default, _ in plain:
            tail = "" if req == "yes" else "  # optional"
            if is_scalar(typ):
                value = default if default != "" else f"<{typ}>"
                lines.append(f'    "{name}": "{value}",{tail}')
            else:
                # A nested value cannot be a form field. Left commented out so
                # the snippet still runs, with the shape named.
                lines.append(f'    # "{name}": json.dumps(...),  # {typ}, '
                             f'send as a JSON string{tail.replace("  #", " —")}')
        lines.append("}")
        lines.append("fields = [(k, (None, v)) for k, v in form.items()]")
        for name, *_ in binaries:
            lines.append(
                f'fields.append(("{name}", ("clip.mp4", open("clip.mp4", "rb"), "video/mp4")))')
        lines.append(f'r = requests.post(f"{{SERVER}}{path}", files=fields, timeout=TIMEOUT)')
    else:
        lines.append("payload = {")
        for name, typ, req, default, _ in plain:
            if not is_scalar(typ):
                value = f"{{}}  # {typ}, fill in from the fields table"
            elif default == "":
                value = '"..."' if typ.startswith("string") else "None"
            elif typ.startswith(("integer", "number")):
                value = default
            elif typ.startswith("boolean"):
                value = "True" if default.lower() == "true" else "False"
            else:
                value = f'"{default}"'
            tail = "" if req == "yes" else "  # optional"
            lines.append(f'    "{name}": {value},{tail}')
        lines.append("}")
        lines.append(f'r = requests.post(f"{{SERVER}}{path}", json=payload, timeout=TIMEOUT)')

    lines.append("print(r.status_code, r.headers.get('content-type'))")
    return "\n".join(lines)


def describe_operation(key):
    """Everything needed to build one request, for the endpoint detail view."""
    spec = SCHEMA["spec"]
    if not key or not spec:
        return "Load the schema first.", [], "", ""
    method, _, path = key.partition(" ")
    op = ((spec.get("paths") or {}).get(path) or {}).get(method.lower())
    if not isinstance(op, dict):
        return f"`{key}` is not in the schema any more — reload it.", [], "", ""
    op = deref(op, spec)

    out = [f"### `{method} {path}`"]
    if op.get("summary"):
        out.append(" ".join(str(op["summary"]).split()))

    ctype, schema = body_schema(op)
    if ctype:
        offered = list(((op.get("requestBody") or {}).get("content") or {}))
        out.append(f"**Request content types:** {', '.join(offered)} — fields below are `{ctype}`.")
        rows = field_rows(schema)
        binaries = [r[0] for r in rows if "binary" in r[1]]
        if binaries:
            out.append("**Binary (file) fields:** " + ", ".join(f"`{b}`" for b in binaries)
                       + " — this is where an image or a video clip goes.")
        else:
            out.append("**Binary (file) fields:** none — this endpoint takes no upload.")
    else:
        rows = [[p.get("name", ""), type_name(p.get("schema") or {}),
                 "yes" if p.get("required") else "", "",
                 " ".join(str(p.get("description") or "").split())[:400]]
                for p in (op.get("parameters") or []) if isinstance(p, dict)]
        out.append("_No request body._" if not rows else "**Query/path parameters:**")

    responses = op.get("responses") or {}
    if responses:
        out.append("**Responses**")
        for code in sorted(responses, key=str):
            content = (responses[code] or {}).get("content") or {}
            out.append(f"- `{code}` → {', '.join(content) if content else '(no body)'}")

    snippet = request_snippet(method, path, ctype or "application/json", rows)
    # An empty Dataframe renders as a blank box that looks like a failure, so
    # say out loud that there is nothing to send.
    rows = rows or [["—", "", "", "", "this operation declares no fields"]]
    raw = json.dumps(op, ensure_ascii=False, indent=2)
    if len(raw) > 60000:
        raw = raw[:60000] + "\n… truncated"
    return "\n\n".join(out), rows, snippet, raw


def walk_properties(spec):
    """(trail, name, subschema) for every property anywhere in the document.

    Walking the whole thing beats guessing which endpoint owns a field: an
    action input could live on the video endpoint, on its own endpoint, or only
    in components/schemas as dead weight.
    """
    found = []

    def walk(node, trail, depth):
        if depth > 24:
            return
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                for name, sub in props.items():
                    found.append((trail, name, sub if isinstance(sub, dict) else {}))
            for k, v in node.items():
                walk(v, f"{trail}/{k}" if trail else str(k), depth + 1)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{trail}[{i}]", depth + 1)

    walk(spec, "", 0)
    return found


def where(trail):
    """A short readable location for a property found by the walker."""
    if trail.startswith("paths/"):
        rest = trail[len("paths/"):]
        for m in ("post", "get", "put", "patch", "delete"):
            marker = f"/{m}/"
            if marker in rest:
                endpoint, _, tail = rest.partition(marker)
                kind = ("request" if "requestBody" in tail
                        else "response" if "responses" in tail else "op")
                return f"{m.upper()} {endpoint} ({kind})"
    parts = trail.split("/")
    if parts[:2] == ["components", "schemas"] and len(parts) > 2:
        return f"schema {parts[2]}"
    return trail[:70] or "(root)"


def flatten_schema(schema, loc, out, prefix="", depth=0):
    """Collect nested property names as dotted paths: data[].action, and so on.

    A field that decides whether a modality is usable is often not top level —
    an action can arrive inside the 200 response's data array — and a top-level-
    only scan would report it as absent.
    """
    if depth > 3 or not isinstance(schema, dict):
        return
    props = schema.get("properties")
    if isinstance(props, dict):
        for name, sub in props.items():
            sub = sub if isinstance(sub, dict) else {}
            out.append((loc, f"{prefix}{name}", sub))
            flatten_schema(sub, loc, out, f"{prefix}{name}.", depth + 1)
    items = schema.get("items")
    if isinstance(items, dict):
        flatten_schema(items, loc, out, f"{prefix}[].", depth + 1)


def operation_properties(spec):
    """(location, name, subschema) for everything reachable from a real operation.

    walk_properties alone is not enough: a body declared as a $ref shows up only
    under components/schemas, so the walker cannot tell a field you can actually
    send to an endpoint from a leftover definition nothing references. deref
    first, then attribute each field to its method and path.
    """
    out = []
    for path, item in (spec.get("paths") or {}).items():
        if not isinstance(item, dict):
            continue
        for method, op in item.items():
            if method.lower() not in ("get", "post", "put", "patch", "delete"):
                continue
            if not isinstance(op, dict):
                continue
            full = deref(op, spec)
            _, schema = body_schema(full)
            flatten_schema(schema, f"{method.upper()} {path} (request)", out)
            for code, resp in (full.get("responses") or {}).items():
                for media in ((resp or {}).get("content") or {}).values():
                    flatten_schema((media or {}).get("schema") or {},
                                   f"{method.upper()} {path} ({code} response)", out)
    return out


def matching_properties(spec, words):
    """Deduplicated (location, name, type) for properties whose name contains a word.

    Operation-level hits are listed first because they are the actionable ones;
    the raw walk is kept as a fallback so an unreferenced definition still shows
    up rather than silently reading as "not supported".
    """
    candidates = operation_properties(spec)
    candidates += [(where(t), n, s) for t, n, s in walk_properties(spec)]
    seen, hits = set(), []
    for loc, name, sub in candidates:
        if not any(w in name.lower() for w in words):
            continue
        key = (loc, name)
        if key in seen:
            continue
        seen.add(key)
        hits.append((loc, name, type_name(sub)))
    return hits


def coverage_report(spec):
    """Answer the four questions that block the unimplemented Table 1 rows.

    These are the questions worth answering before writing any of the missing
    tabs, and the answers change per build, so they are computed rather than
    written down.
    """
    paths = set(spec.get("paths") or {})
    out = ["#### Table 1 coverage probe"]

    # Row 2 — video in, video out. The GUI already posts multipart to the video
    # endpoint; the only question is whether its binary slot accepts an mp4.
    video_op = ((spec.get("paths") or {}).get(VIDEO_PATH) or {}).get("post")
    if not video_op:
        out.append(f"**Text|Video → Video** — `{VIDEO_PATH}` is not served at all. "
                   "Nothing to extend; find the right path in the list below.")
    else:
        _, vschema = body_schema(deref(video_op, spec))
        vrows = field_rows(vschema)
        bins = [r for r in vrows if "binary" in r[1]]
        if not bins:
            out.append(f"**Text|Video → Video** — `{VIDEO_PATH}` declares no binary field, "
                       "so this build looks text-only. Check the raw JSON before believing it.")
        else:
            media = [r for r in bins if "video" in (r[4] or "").lower()]
            names = ", ".join(f"`{r[0]}`" for r in bins)
            if media:
                out.append(f"**Text|Video → Video** — binary slots {names}; "
                           f"`{media[0][0]}` mentions video in its schema. Add a `gr.Video` "
                           "input and reuse `post_video()` as is.")
            else:
                out.append(f"**Text|Video → Video** — binary slots {names}, but the schema "
                           "does not say which media types they accept. `post_video()` already "
                           "guesses the mimetype from the filename, so send one mp4 and read "
                           "the 422 if it is refused.")

    # Row 5 — action in.
    act = matching_properties(spec, ACTION_WORDS)
    act_req = [h for h in act if "request" in h[0]]
    if act_req:
        out.append("**Action|Video|Text → Video** — action fields on request bodies: "
                   + "; ".join(f"`{n}` ({t}) on {loc}" for loc, n, t in act_req[:6]))
    elif act:
        out.append("**Action|Video|Text → Video** — the word *action* appears only outside "
                   "request bodies (" + ", ".join(f"`{n}` on {loc}" for loc, n, _ in act[:4])
                   + "), so there may be no way to feed actions in on this build.")
    else:
        out.append("**Action|Video|Text → Video** — no field named *action* anywhere in the "
                   "document. Not implementable against this server as it stands.")

    # Row 4 — text out. Separate endpoint, so presence is a clean yes/no.
    chat = [p for p in CHAT_PATHS if p in paths]
    if chat:
        out.append("**Text|Image|Video → Text** — served at "
                   + ", ".join(f"`{p}`" for p in chat)
                   + ". Standard OpenAI chat shape, so a VLM tab is self-contained.")
    else:
        guesses = sorted(p for p in paths if "chat" in p or "completion" in p or "generate" in p)
        out.append("**Text|Image|Video → Text** — none of "
                   + ", ".join(f"`{p}`" for p in CHAT_PATHS) + " is served."
                   + (" Closest: " + ", ".join(f"`{p}`" for p in guesses[:4]) if guesses else ""))

    # Row 6 — action out. Response parsing is the blocker, not the request.
    act_resp = [h for h in act if "response" in h[0]]
    if act_resp:
        out.append("**Video|Text → Video|Action** — action fields on responses: "
                   + "; ".join(f"`{n}` ({t}) on {loc}" for loc, n, t in act_resp[:6])
                   + ". `post_video()` only recognises mp4 bytes today, so this needs a "
                   "second response branch.")
    else:
        out.append("**Video|Text → Video|Action** — no action field on any response schema. "
                   "Either this build cannot emit actions, or it hides them in a free-form "
                   "field — check the video endpoint's 200 response below.")

    robot = matching_properties(spec, ROBOT_WORDS)
    if robot:
        out.append("_Other robot-shaped fields worth reading:_ "
                   + ", ".join(f"`{n}` on {loc}" for loc, n, _ in robot[:8]))
    return "\n\n".join(out)


def overview(force=True):
    """Everything the schema view needs, as plain data.

    A dict rather than a tuple of gradio updates: this module must not know that
    a UI exists, so the caller decides how to render it.

    Keys: error, summary (markdown), operations [(label, key)], preselect, raw.
    """
    spec, err = fetch_spec(force=force)
    if not spec:
        return {
            "error": err,
            "summary": (f"**{err}**\n\nThe container is probably still loading the "
                        "model — start-up takes a while. Try again."),
            "operations": [],
            "preselect": None,
            "raw": "",
        }

    info = spec.get("info") or {}
    paths = spec.get("paths") or {}
    ops = operation_choices(spec)
    summary = "\n\n".join([
        f"**{info.get('title', 'server')} {info.get('version', '')}** at `{SERVER}` — "
        f"{len(paths)} paths, {len(ops)} operations. Read at {SCHEMA['fetched']}.",
        coverage_report(spec),
    ])
    raw = json.dumps(spec, ensure_ascii=False, indent=2)
    if len(raw) > 200000:
        raw = raw[:200000] + "\n… truncated, save a snapshot for the whole thing"
    # Preselect the video endpoint: every remaining modality goes through it, so
    # it is what you actually came here to read.
    preselect = next((k for _, k in ops if k.endswith(VIDEO_PATH)),
                     ops[0][1] if ops else None)
    return {"error": "", "summary": summary, "operations": ops,
            "preselect": preselect, "raw": raw}


def save_snapshot(out_dir="."):
    """Write the schema to disk. Returns a status line.

    Worth keeping: when a container image is updated the field names can move,
    and an old snapshot is the only way to tell what changed. The cosmos_ prefix
    with a .json suffix is deliberately excluded from the GUI's History list.
    """
    spec, err = fetch_spec(force=True)
    if not spec:
        return err
    path = os.path.join(out_dir, f"cosmos_openapi_{int(time.time())}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    return f"Saved to {path}"


def strip_markdown(text):
    """Markdown is for the GUI; a terminal wants the same words without the noise.

    Underscores are removed only at word edges. Stripping them everywhere turns
    input_reference into inputreference, which is worse than useless in a report
    whose whole job is telling you the exact field name.
    """
    for token in ("####", "###", "**", "`"):
        text = text.replace(token, "")
    return re.sub(r"(?<!\w)_|_(?!\w)", "", text).strip()


def main(argv):
    """The CLI. Exit code 1 when the server could not be read, so this can gate
    a shell script that waits for the container to finish loading."""
    if argv[:1] == ["--save"]:
        print(save_snapshot(argv[1] if len(argv) > 1 else "."))
        return 0

    if argv:
        # "POST /v1/videos/sync", quoted or not.
        key = " ".join(argv)
        if not fetch_spec()[0]:
            print(fetch_spec()[1], file=sys.stderr)
            return 1
        info, rows, snippet, raw = describe_operation(key)
        print(strip_markdown(info))
        print()
        wn = max([len(r[0]) for r in rows] + [len("field")])
        wt = max([len(r[1]) for r in rows] + [len("type")])
        print(f"{'field'.ljust(wn)}  {'type'.ljust(wt)}  req   default")
        print("-" * (wn + wt + 20))
        for name, typ, req, default, notes in rows:
            print(f"{name.ljust(wn)}  {typ.ljust(wt)}  {req or '-':<5} {default or '-'}")
            if notes:
                print(f"{' ' * wn}  {notes[:110]}")
        print("\n--- snippet " + "-" * 40)
        print(snippet)
        return 0

    report = overview()
    print(strip_markdown(report["summary"]))
    if report["error"]:
        return 1
    print("\n--- endpoints " + "-" * 40)
    for label, _ in report["operations"]:
        print(" ", label)
    print("\nDetail:  python cosmos_schema.py POST " + VIDEO_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
