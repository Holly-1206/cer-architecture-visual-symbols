# -*- coding: utf-8 -*-
"""
Stage-2 preparation for the Chinese Eastern Railway LabelMe corpus.

pip install pillow pandas

This script:
1. scans all LabelMe JSON files;
2. reconstructs missing images from LabelMe imageData when possible;
3. normalizes only safe label-format variants (b4->B4, A7/->A7);
4. keeps semantic labels separate and flags suspicious labels;
5. computes full-corpus pixel hashes and dHash;
6. reports cross-style duplicates;
7. writes a master manifest for later model training.

It NEVER edits the original dataset.
"""

import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("CER_PROJECT_ROOT", Path(__file__).resolve().parents[1]))  # set CER_PROJECT_ROOT or keep the repository layout
import base64, hashlib, io, json, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps

ROOTS = [
    ("historical_album", Path(str(PROJECT_ROOT / "annotations" / "historical_album"))),
    ("expanded_reference", Path(str(PROJECT_ROOT / "annotations" / "expanded_reference"))),
    ("field_photographs", Path(str(PROJECT_ROOT / "annotations" / "field_photographs"))),
]
OUT = Path(str(PROJECT_ROOT / "prepared_corpus_v2"))
IMG_OUT = OUT / "images"

STYLE_NAMES = {
    "01":"俄罗斯传统建筑","02":"俄式木构民居","03":"俄罗斯铁路建筑",
    "04":"拜占庭_东正教风格","05":"折衷主义","06":"新艺术运动",
    "07":"巴洛克","08":"中华巴洛克","09":"古典主义_新古典",
    "10":"哥特式","11":"装饰艺术","12":"早期现代主义",
}
STYLE_RE = re.compile(r"^\s*(0[1-9]|1[0-2])(?=[_\s\u4e00-\u9fff])")
IMG_EXTS = [".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"]

def norm(x):
    s = unicodedata.normalize("NFKC", "" if x is None else str(x))
    return re.sub(r"\s+", " ", s).strip()

def canon_label(x):
    s = norm(x).rstrip("\\/").strip()
    m = re.fullmatch(r"([A-Ea-e])(\d+)", s)
    if m:
        return m.group(1).upper()+m.group(2), "coded_AE"
    if s.startswith(("WIN_","WALL_","AUX_","STRUCT_","ROOF_","DEC_","SPACE_","DOOR_","MAT_")):
        return s, "semantic_named"
    return s, "review_required"

def infer_style(path, root):
    cur = path.parent
    while True:
        m = STYLE_RE.match(cur.name)
        if m:
            sid = m.group(1)
            return sid, STYLE_NAMES[sid]
        if cur == root or cur.parent == cur:
            break
        cur = cur.parent
    return "UNRESOLVED","UNRESOLVED"

def read_json(path):
    last = ""
    for enc in ("utf-8-sig","utf-8","gb18030"):
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f), enc, ""
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
    return None, "", last

def find_external(json_path, data):
    ip = data.get("imagePath")
    candidates = []
    if ip:
        p = Path(str(ip).replace("\\","/"))
        candidates += [p if p.is_absolute() else json_path.parent/p, json_path.parent/p.name]
    candidates += [json_path.with_suffix(ext) for ext in IMG_EXTS]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None

def recover_image(json_path, data):
    p = find_external(json_path, data)
    if p:
        try:
            with Image.open(p) as im:
                return ImageOps.exif_transpose(im).convert("RGB"), "external_file", ""
        except Exception as e:
            ext_err = f"{type(e).__name__}: {e}"
    else:
        ext_err = ""
    payload = data.get("imageData")
    if isinstance(payload, str) and payload.strip():
        try:
            payload = payload.strip()
            if payload.lower().startswith("data:image") and "," in payload:
                payload = payload.split(",",1)[1]
            raw = base64.b64decode(payload, validate=False)
            with Image.open(io.BytesIO(raw)) as im:
                return ImageOps.exif_transpose(im).convert("RGB"), "embedded_imageData", ""
        except Exception as e:
            return None, "unresolved", f"imageData: {type(e).__name__}: {e}"
    return None, "unresolved", ext_err or "No external file and no usable imageData."

def pixel_hash(im):
    im = im.convert("RGB")
    h = hashlib.sha256()
    h.update(f"{im.width}x{im.height}|".encode())
    h.update(im.tobytes())
    return h.hexdigest()

def dhash(im, n=8):
    g = im.convert("L").resize((n+1,n))
    px = list(g.getdata())
    bits = []
    for y in range(n):
        off = y*(n+1)
        for x in range(n):
            bits.append(px[off+x] > px[off+x+1])
    v=0
    for b in bits: v=(v<<1)|int(b)
    return f"{v:016x}"

def safe_name(s):
    return re.sub(r'[<>:"/\\|?*]+', "_", s).strip(" .")[:150] or "unnamed"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    IMG_OUT.mkdir(parents=True, exist_ok=True)

    manifest, review, unresolved = [], [], []
    label_shapes, label_images = Counter(), Counter()
    label_src, label_style, shape_types = defaultdict(Counter), defaultdict(Counter), Counter()

    for source, root in ROOTS:
        if not root.exists():
            print("[MISSING]", root)
            continue
        js = sorted(root.rglob("*.json"))
        print(f"{source}: {len(js)} JSON")
        for i,jp in enumerate(js,1):
            sid,sname = infer_style(jp,root)
            data,enc,err = read_json(jp)
            if not isinstance(data,dict):
                unresolved.append([source,sid,str(jp),"malformed_json",err])
                continue

            shapes = data.get("shapes",[])
            if not isinstance(shapes,list): shapes=[]
            labels=[]
            for k,sh in enumerate(shapes):
                if not isinstance(sh,dict): continue
                raw = sh.get("label","")
                lab,ns = canon_label(raw)
                labels.append(lab)
                label_shapes[lab]+=1
                label_src[lab][source]+=1
                label_style[lab][sid]+=1
                shape_types[norm(sh.get("shape_type") or "polygon")]+=1
                if ns=="review_required":
                    review.append([source,sid,str(jp),k,norm(raw),lab])
            for lab in set(labels):
                label_images[lab]+=1

            im,origin,ierr = recover_image(jp,data)
            outpath=""; ph=""; dh=""; w=""; h=""
            if im is not None:
                w,h=im.size
                ph=pixel_hash(im); dh=dhash(im)
                od=IMG_OUT/source/sid; od.mkdir(parents=True,exist_ok=True)
                op=od/f"{safe_name(jp.stem)}__{ph[:10]}.jpg"
                if not op.exists(): im.save(op,"JPEG",quality=95)
                outpath=str(op)
            else:
                unresolved.append([source,sid,str(jp),"unresolved_image",ierr])

            manifest.append({
                "source":source,"style_id":sid,"style_name":sname,
                "json_path":str(jp),"encoding":enc,
                "has_imageData":bool(isinstance(data.get("imageData"),str) and data.get("imageData").strip()),
                "image_origin":origin,"extracted_image_path":outpath,
                "width":w,"height":h,"pixel_sha256":ph,"dhash":dh,
                "shape_count":len(shapes),"labels":" | ".join(sorted(set(labels))),
            })
            if i%250==0: print(f"  {i}/{len(js)}")

    df=pd.DataFrame(manifest)
    df.to_csv(OUT/"01_master_manifest.csv",index=False,encoding="utf-8-sig")

    inv=[]
    for lab,n in label_shapes.items():
        _,ns=canon_label(lab)
        inv.append({
            "label":lab,"namespace":ns,"shape_instances":n,
            "annotated_images":label_images[lab],
            "sources":" | ".join(f"{k}:{v}" for k,v in sorted(label_src[lab].items())),
            "styles":" | ".join(f"{k}:{v}" for k,v in sorted(label_style[lab].items())),
        })
    pd.DataFrame(inv).sort_values("shape_instances",ascending=False).to_csv(
        OUT/"02_label_inventory_clean.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(review,columns=["source","style_id","json_path","shape_index","raw_label","canonical_label"]).to_csv(
        OUT/"03_label_review_required.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(shape_types.most_common(),columns=["shape_type","count"]).to_csv(
        OUT/"04_shape_type_summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(unresolved,columns=["source","style_id","json_path","problem","details"]).to_csv(
        OUT/"07_unresolved_images.csv",index=False,encoding="utf-8-sig")

    recovered=df[df["pixel_sha256"].astype(str)!=""].copy()
    dups=[]; cross=[]
    gid=0
    for ph,g in recovered.groupby("pixel_sha256"):
        if len(g)<2: continue
        gid+=1
        nstyles=g.style_id.nunique(); nsources=g.source.nunique()
        for _,r in g.iterrows():
            row=[gid,ph,len(g),nstyles,nsources,r.source,r.style_id,r.style_name,r.json_path,r.extracted_image_path]
            dups.append(row)
            if nstyles>1: cross.append(row)
    cols=["duplicate_group","pixel_sha256","group_size","n_styles","n_sources","source","style_id","style_name","json_path","extracted_image_path"]
    pd.DataFrame(dups,columns=cols).to_csv(OUT/"05_duplicate_groups_all.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(cross,columns=cols).to_csv(OUT/"06_cross_style_duplicates.csv",index=False,encoding="utf-8-sig")

    ss=(df.groupby(["source","style_id","style_name"])
          .agg(json_records=("json_path","size"),
               recovered_images=("extracted_image_path",lambda x:(x.astype(str)!="").sum()),
               empty_annotations=("shape_count",lambda x:(x==0).sum()),
               total_shapes=("shape_count","sum")).reset_index())
    ss.to_csv(OUT/"08_source_style_counts.csv",index=False,encoding="utf-8-sig")

    origin=df.image_origin.value_counts().to_dict()
    print("\nDONE")
    print("records:",len(df))
    print("recovered:",(df.extracted_image_path.astype(str)!="").sum())
    print("origins:",origin)
    print("duplicate groups:",len(set(x[0] for x in dups)))
    print("cross-style duplicate groups:",len(set(x[0] for x in cross)))
    print("output:",OUT)

if __name__=="__main__":
    main()
